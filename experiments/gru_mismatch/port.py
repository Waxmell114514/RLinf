"""Faithful port of the exp3/exp4 GRU protocol from "From Prediction to Self".

Source: ``yefeifei-tech/prediction-to-self-journal`` (MIT, see
``LICENSE-prediction-to-self``), files ``core/world.py``,
``experiments/exp3_encoding_gap.py`` and ``experiments/exp4_proprioception.py``.

The port keeps Evan Ye's constants, module construction order and random-number
consumption, so ``interface="none"`` is exp3 and ``interface="trace_abs"`` is
exp4 step for step.  The only additions are two further efference interfaces
(below) and the hooks :mod:`.mismatch` needs to keep running the trained model
frozen.  Two properties of the original are worth knowing before reading any
number this produces:

* **Phase 1 never trains the GRU.**  ``step_frozen`` predicts from the
  *previous* ``h_multi``, which is detached, so the MSE gradient reaches only
  the ``pred`` readout.  Phase 1 fits a readout on a random recurrent
  reservoir.
* **Phase 2 trains the GRU only through the auxiliary 3-class head**
  (active / trailing / quiet), with one-step truncated gradients.  "Trailing"
  -- the window right after a burst of actions ends -- is a supervised label
  there, and the reported recall is the *online* recall of a probe trained
  concurrently, counted over all of phase 2.

Efference interfaces (``interface``):

``none``          exp3: GRU input is the 4-d observation only.
``trace_abs``     exp4: observation + ``tau <- beta*tau + (1-beta)*|a|``.
``trace_signed``  as exp4 but the trace keeps the sign of the command.
``command``       observation + the instantaneous signed command ``a_t``.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

INTERFACES = ("none", "trace_abs", "trace_signed", "command")

CLASS_ACTIVE = 0
CLASS_TRAILING = 1
CLASS_QUIET = 2

# Evan Ye's `--quick` divides these step counts by 5.
REFERENCE_SEED = 42
REFERENCE_BURST_SEED = 456


@dataclass
class ProtocolConfig:
    """exp3/exp4 constants, verbatim, plus the efference interface."""

    interface: str = "trace_abs"
    seed: int = REFERENCE_SEED
    # 1 = the paper's protocol; 5 = the original `--quick` smoke mode.
    scale: int = 1
    train_steps: int = 100_000
    phase2_steps: int = 80_000
    lr_p1: float = 1e-3
    lr_p2: float = 1e-4
    aux_lr: float = 5e-4
    probe_lr: float = 5e-4
    gamma: float = 2.0
    action_range: float = 2.0
    trace_beta: float = 0.95
    aux_weight: float = 0.1
    trailing_class_weight: float = 5.0
    trailing_steps: int = 50
    burst_mean: int = 400
    burst_min: int = 100
    burst_max: int = 1000
    hidden_dim: int = 192
    n_scales: int = 4
    obs_dim: int = 4
    log_every: int = 20_000

    def __post_init__(self) -> None:
        if self.interface not in INTERFACES:
            raise ValueError(
                f"unknown interface {self.interface!r}; use one of {INTERFACES}"
            )
        if self.scale < 1:
            raise ValueError("scale must be >= 1")

    def steps(self, value: int) -> int:
        """Apply the quick-mode divisor the way the original does (floor)."""
        return value // self.scale

    @property
    def burst_seed(self) -> int:
        # The original pins the gate to 456; other seeds shift it so that the
        # burst schedule is a seed-level replicate too, not a shared constant.
        return REFERENCE_BURST_SEED + (self.seed - REFERENCE_SEED)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SineSignal:
    """``core/world.py::SineSignal``: 4-channel sum of sinusoids plus noise.

    Draws its noise from the *global* numpy RNG, exactly like the original;
    that shared stream is what makes the action draws line up with the source.
    """

    CHANNELS = (
        ((0.05, 0.13), (1.0, 0.5)),
        ((0.08, 0.21), (0.8, 0.4)),
        ((0.03, 0.17), (0.9, 0.3)),
        ((0.11, 0.29), (0.7, 0.6)),
    )
    NOISE = 0.05
    DT = 0.01

    def __init__(self) -> None:
        self.t = 0

    def get(self) -> torch.Tensor:
        moment = self.t * self.DT
        values = []
        for freqs, amps in self.CHANNELS:
            value = sum(a * np.sin(2 * np.pi * f * moment) for f, a in zip(freqs, amps))
            value += np.random.normal(0, self.NOISE)
            values.append(value)
        self.t += 1
        return torch.tensor(values, dtype=torch.float32)


class BurstGate:
    """Alternating active/quiet periods with clipped exponential lengths."""

    def __init__(
        self, mean_p: int = 400, min_p: int = 100, max_p: int = 1000, seed: int = 456
    ) -> None:
        self.rng = random.Random(seed)
        self.mean_p = mean_p
        self.min_p = min_p
        self.max_p = max_p
        self.active = False
        self.next_switch = 0
        self._prev = False

    def step(self, t: int) -> bool:
        self._prev = self.active
        if t >= self.next_switch:
            self.active = not self.active
            interval = int(self.rng.expovariate(1.0 / self.mean_p))
            self.next_switch = t + max(self.min_p, min(interval, self.max_p))
        return self.active

    @property
    def just_off(self) -> bool:
        return self._prev and not self.active

    @property
    def just_on(self) -> bool:
        return self.active and not self._prev


class GRUAgent(nn.Module):
    """exp3 ``Model`` / exp4 ``ProprioceptiveModel`` with a pluggable efference input.

    Construction order (GRU, then readout) matches the source, so the torch
    RNG hands out the same initial weights.
    """

    def __init__(
        self,
        interface: str = "trace_abs",
        obs_dim: int = 4,
        hidden_dim: int = 192,
        n_scales: int = 4,
        trace_beta: float = 0.95,
    ) -> None:
        super().__init__()
        if interface not in INTERFACES:
            raise ValueError(f"unknown interface {interface!r}")
        self.interface = interface
        self.hidden_dim = hidden_dim
        self.trace_beta = trace_beta
        extra = 0 if interface == "none" else 1
        self.gru = nn.GRUCell(obs_dim + extra, hidden_dim)
        self.pred = nn.Linear(hidden_dim, obs_dim)
        alphas = torch.logspace(math.log10(0.02), math.log10(0.80), n_scales)
        per_group = hidden_dim // n_scales
        alpha_vec = torch.cat([a.expand(per_group) for a in alphas])
        if len(alpha_vec) < hidden_dim:
            alpha_vec = torch.cat(
                [alpha_vec, alpha_vec[-1:].expand(hidden_dim - len(alpha_vec))]
            )
        self.register_buffer("alpha", alpha_vec.unsqueeze(0))
        self.h_multi = torch.zeros(1, hidden_dim)
        self.h_gru = torch.zeros(1, hidden_dim)
        # The |a| trace is tracked for every interface -- it is the probe
        # control C_eff -- but only fed to the GRU when the interface says so.
        self.trace = 0.0
        self.command = 0.0

    # -- efference -------------------------------------------------------

    def observe_command(self, action_val: float) -> None:
        """Record the outgoing *command* (before any hijack touches the world)."""
        magnitude = action_val if self.interface == "trace_signed" else abs(action_val)
        self.trace = self.trace_beta * self.trace + (1 - self.trace_beta) * magnitude
        self.command = float(action_val)

    @property
    def efference_value(self) -> float:
        if self.interface == "command":
            return self.command
        return self.trace

    def _gru_input(self, obs: torch.Tensor) -> torch.Tensor:
        if self.interface == "none":
            return obs.unsqueeze(0)
        extra = torch.tensor([[self.efference_value]], dtype=torch.float32)
        return torch.cat([obs.unsqueeze(0), extra], dim=1)

    # -- stepping (verbatim semantics) -----------------------------------

    def step_frozen(self, obs: torch.Tensor) -> torch.Tensor:
        pred = self.pred(self.h_multi)
        h_new = self.gru(self._gru_input(obs), self.h_gru)
        self.h_multi = ((1 - self.alpha) * self.h_multi + self.alpha * h_new).detach()
        self.h_gru = h_new.detach()
        return pred.squeeze(0)

    def step_live(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred = self.pred(self.h_multi)
        h_new = self.gru(self._gru_input(obs), self.h_gru)
        h_live = (1 - self.alpha) * self.h_multi + self.alpha * h_new
        self.h_gru = h_new.detach()
        return pred.squeeze(0), h_live

    def commit(self, h_live: torch.Tensor) -> None:
        self.h_multi = h_live.detach()

    def get_h(self) -> torch.Tensor:
        return self.h_multi.detach().squeeze().clone()


class AuxHead(nn.Module):
    def __init__(self, hidden_dim: int = 192) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


class BinaryProbe(nn.Module):
    def __init__(self, hidden_dim: int = 192) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h).squeeze(-1)


@dataclass
class TrainedAgent:
    """Everything phase 3 needs to continue the same world, frozen."""

    protocol: ProtocolConfig
    model: GRUAgent
    aux: AuxHead
    burst: BurstGate
    signal: SineSignal
    next_step: int
    trailing_remaining: int
    trailing_correct: int
    trailing_total: int
    wall_seconds: float

    @property
    def trailing_recall(self) -> float:
        """The source's headline number, in percent."""
        return 100.0 * self.trailing_correct / max(1, self.trailing_total)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(protocol: ProtocolConfig) -> TrainedAgent:
    """Run phases 1-2 exactly as exp3/exp4 do and return the trained agent."""
    seed_everything(protocol.seed)
    signal = SineSignal()
    model = GRUAgent(
        protocol.interface,
        obs_dim=protocol.obs_dim,
        hidden_dim=protocol.hidden_dim,
        n_scales=protocol.n_scales,
        trace_beta=protocol.trace_beta,
    )
    aux = AuxHead(protocol.hidden_dim)
    probe = BinaryProbe(protocol.hidden_dim)
    burst = BurstGate(
        protocol.steps(protocol.burst_mean),
        protocol.steps(protocol.burst_min),
        protocol.steps(protocol.burst_max),
        seed=protocol.burst_seed,
    )

    train_steps = protocol.steps(protocol.train_steps)
    phase2_steps = protocol.steps(protocol.phase2_steps)
    trailing_steps = protocol.steps(protocol.trailing_steps)
    log_every = max(1, protocol.steps(protocol.log_every))

    opt_p1 = torch.optim.Adam(model.parameters(), lr=protocol.lr_p1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_p1, T_max=train_steps, eta_min=1e-5
    )
    class_weights = torch.tensor([1.0, protocol.trailing_class_weight, 1.0])
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    bce_loss = nn.BCEWithLogitsLoss()
    opt_p2: Optional[torch.optim.Optimizer] = None
    opt_probe: Optional[torch.optim.Optimizer] = None

    start = time.time()
    trailing_remaining = 0
    trailing_correct = 0
    trailing_total = 0
    total_steps = train_steps + phase2_steps
    for step in range(total_steps):
        is_p1 = step < train_steps
        obs_raw = signal.get()

        if not is_p1:
            active = burst.step(step)
            if burst.just_off:
                trailing_remaining = trailing_steps
        else:
            active = True

        # Same draw pattern as the source: one uniform whenever acting.
        if active or is_p1:
            action_val = float(
                np.random.uniform(-protocol.action_range, protocol.action_range)
            )
        else:
            action_val = 0.0

        is_trailing = False
        cur_class = CLASS_ACTIVE
        if not is_p1:
            if trailing_remaining > 0:
                cur_class = CLASS_TRAILING
                is_trailing = True
                trailing_remaining -= 1
            elif active:
                cur_class = CLASS_ACTIVE
            else:
                cur_class = CLASS_QUIET

        # Efference is updated BEFORE the state update, as in exp4.
        model.observe_command(action_val if (active or is_p1) else 0.0)

        obs = obs_raw.clone()
        gamma_eff = protocol.gamma if (active or is_p1) else 0.0
        obs[0] = obs[0] + gamma_eff * action_val

        if is_p1:
            pred = model.step_frozen(obs)
            loss = nn.functional.mse_loss(pred, obs)
            opt_p1.zero_grad()
            loss.backward()
            opt_p1.step()
            sched.step()
            if step > 0 and step % log_every == 0:
                logger.info(
                    "[%s seed %d] p1 step %d mse=%.6f (%.0fs)",
                    protocol.interface,
                    protocol.seed,
                    step,
                    loss.item(),
                    time.time() - start,
                )
            continue

        if opt_p2 is None:
            opt_p2 = torch.optim.Adam(
                [
                    {"params": model.gru.parameters(), "lr": protocol.lr_p2},
                    {"params": model.pred.parameters(), "lr": protocol.lr_p2},
                    {"params": aux.parameters(), "lr": protocol.aux_lr},
                ]
            )
            opt_probe = torch.optim.Adam(probe.parameters(), lr=protocol.probe_lr)

        pred, h_live = model.step_live(obs)
        loss_pred = nn.functional.mse_loss(pred, obs)
        logits = aux(h_live)
        target = torch.tensor([cur_class], dtype=torch.long)
        loss_aux = protocol.aux_weight * ce_loss(logits, target)
        opt_p2.zero_grad()
        (loss_pred + loss_aux).backward()
        opt_p2.step()
        model.commit(h_live)

        logit = probe(model.get_h().unsqueeze(0))
        probe_target = torch.tensor([1.0 if is_trailing else 0.0])
        probe_loss = bce_loss(logit, probe_target)
        opt_probe.zero_grad()
        probe_loss.backward()
        opt_probe.step()

        if is_trailing:
            with torch.no_grad():
                predicted = (torch.sigmoid(logit) > 0.5).float().item()
            trailing_total += 1
            trailing_correct += int(predicted == 1.0)

        if step > 0 and step % log_every == 0:
            logger.info(
                "[%s seed %d] p2 step %d mse=%.6f trailing=%.1f%% (%.0fs)",
                protocol.interface,
                protocol.seed,
                step,
                loss_pred.item(),
                100.0 * trailing_correct / max(1, trailing_total),
                time.time() - start,
            )

    model.eval()
    aux.eval()
    return TrainedAgent(
        protocol=protocol,
        model=model,
        aux=aux,
        burst=burst,
        signal=signal,
        next_step=total_steps,
        trailing_remaining=trailing_remaining,
        trailing_correct=trailing_correct,
        trailing_total=trailing_total,
        wall_seconds=time.time() - start,
    )


__all__ = [
    "CLASS_ACTIVE",
    "CLASS_QUIET",
    "CLASS_TRAILING",
    "INTERFACES",
    "BurstGate",
    "GRUAgent",
    "ProtocolConfig",
    "SineSignal",
    "TrainedAgent",
    "seed_everything",
    "train",
]
