"""Phase 3: run a trained exp3/exp4 agent frozen, with hijacked actions.

The mismatch condition of the line-A plan: the efference input receives the
*command* ``a`` while the world moves by ``gamma * a'``, ``a' != a``.  This is
the GRU analogue of the VLA intervention study, and the transforms are the
same three:

``swap``    ``a'`` is an independent draw from the command distribution (the
            VLA's "another episode's chunk": same marginal statistics).
``mirror``  ``a' = -a``.
``freeze``  ``a' = 0`` (the world does not respond to the command).

Like the SFT VLA, the agent never sees a hijack during training; phase 3 only
reads it out.  Nothing is updated in phase 3.

All three arms replay the *same* world: the global numpy stream (signal noise
and commands), the burst gate and the hijack schedule are re-seeded
identically per arm, and the swap draw comes from its own stream that every
arm consumes.  The arms therefore differ in the executed action on hijacked
steps and in nothing else -- a paired design across transforms.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from .port import CLASS_ACTIVE, CLASS_QUIET, CLASS_TRAILING, TrainedAgent

TRANSFORMS = ("swap", "mirror", "freeze")
HIJACK = "HIJACK"
SELF = "SELF"


@dataclass
class MismatchConfig:
    """Phase-3 schedule; mirrors the VLA hijack schedule where it can."""

    steps: int = 40_000
    p_hijack: float = 0.25
    # Steps since burst onset before a hijack may land (VLA: 4-call warm-up).
    warmup: int = 4
    # Clean steps required after a hijack (VLA: min_clean_gap = 2).
    min_clean_gap: int = 2
    # Offset for the phase-3 streams, so they never alias training's seed.
    seed_offset: int = 7919

    def __post_init__(self) -> None:
        if self.min_clean_gap < 1:
            raise ValueError("min_clean_gap must be >= 1")
        if not 0.0 <= self.p_hijack <= 1.0:
            raise ValueError("p_hijack out of range")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def execute(transform: str, command: float, donor: float) -> float:
    """The action the world actually applies on a hijacked step."""
    if transform == "swap":
        return float(donor)
    if transform == "mirror":
        return -float(command)
    if transform == "freeze":
        return 0.0
    raise ValueError(f"unknown transform {transform!r}; use {TRANSFORMS}")


@dataclass
class ArmData:
    """One phase-3 arm: per-step scalars plus the hidden states."""

    transform: str
    steps: pd.DataFrame
    h_multi: np.ndarray  # [n_steps, hidden] float32, state AFTER the step
    h_gru: np.ndarray  # [n_steps, hidden] float32


def collect(trained: TrainedAgent, transform: str, config: MismatchConfig) -> ArmData:
    """Run ``config.steps`` frozen steps under one hijack transform."""
    if transform not in TRANSFORMS:
        raise ValueError(f"unknown transform {transform!r}; use {TRANSFORMS}")
    protocol = trained.protocol
    model = copy.deepcopy(trained.model)
    burst = copy.deepcopy(trained.burst)
    signal = copy.deepcopy(trained.signal)
    trailing_remaining = trained.trailing_remaining
    trailing_steps = protocol.steps(protocol.trailing_steps)

    base_seed = protocol.seed + config.seed_offset
    np.random.seed(base_seed)
    schedule_rng = np.random.default_rng([base_seed, 1])
    donor_rng = np.random.default_rng([base_seed, 2])

    n = config.steps
    hidden = protocol.hidden_dim
    h_multi = np.zeros((n, hidden), dtype=np.float32)
    h_gru = np.zeros((n, hidden), dtype=np.float32)
    rows: list[dict[str, Any]] = []

    # A burst already running when phase 3 starts has an unknown onset, so it
    # gets segment 0 and is excluded from samples downstream.
    segment = 0
    since_onset = -1
    last_hijack = -(10**9)
    with torch.no_grad():
        for index in range(n):
            step = trained.next_step + index
            obs_raw = signal.get()
            active = burst.step(step)
            if burst.just_off:
                trailing_remaining = trailing_steps
            if burst.just_on:
                segment += 1
                since_onset = 0
            elif active and since_onset >= 0:
                since_onset += 1

            command = (
                float(np.random.uniform(-protocol.action_range, protocol.action_range))
                if active
                else 0.0
            )
            if trailing_remaining > 0:
                cls = CLASS_TRAILING
                trailing_remaining -= 1
            elif active:
                cls = CLASS_ACTIVE
            else:
                cls = CLASS_QUIET

            # Both streams advance every step, hijack or not, so the schedule
            # and the donor draws are identical across arms.
            draw = schedule_rng.random()
            donor = float(
                donor_rng.uniform(-protocol.action_range, protocol.action_range)
            )
            eligible = (
                active
                and since_onset >= config.warmup
                and (index - last_hijack) > config.min_clean_gap
            )
            hijacked = bool(eligible and draw < config.p_hijack)
            if hijacked:
                last_hijack = index
                executed = execute(transform, command, donor)
            else:
                executed = command

            model.observe_command(command)
            obs = obs_raw.clone()
            gamma_eff = protocol.gamma if active else 0.0
            obs[0] = obs[0] + gamma_eff * executed

            pred = model.pred(model.h_multi).squeeze(0)
            model.step_frozen(obs)
            h_multi[index] = model.h_multi.squeeze(0).numpy()
            h_gru[index] = model.h_gru.squeeze(0).numpy()
            residual = (obs - pred).numpy()
            rows.append(
                {
                    "index": index,
                    "segment": segment,
                    "since_onset": since_onset if active else -1,
                    "active": bool(active),
                    "cls": cls,
                    "command": command,
                    "executed": float(executed),
                    "label": HIJACK if hijacked else SELF,
                    "transform": transform if hijacked else "",
                    "trace": float(model.trace),
                    "efference": float(model.efference_value)
                    if protocol.interface != "none"
                    else 0.0,
                    **{f"obs{k}": float(obs[k]) for k in range(obs.shape[0])},
                    **{f"res{k}": float(residual[k]) for k in range(residual.shape[0])},
                }
            )
    return ArmData(
        transform=transform,
        steps=pd.DataFrame(rows),
        h_multi=h_multi,
        h_gru=h_gru,
    )


__all__ = [
    "HIJACK",
    "SELF",
    "TRANSFORMS",
    "ArmData",
    "MismatchConfig",
    "collect",
    "execute",
]
