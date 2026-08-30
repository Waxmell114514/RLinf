"""Hijack transforms and schedule.

Pure numpy: no torch, no LIBERO.  Everything here is unit-tested offline so the
GPU run does not spend its budget discovering scheduling bugs.

All transforms operate in the *environment* action space, i.e. after
``rlinf.envs.action_utils.prepare_actions`` has mapped the policy's gripper
output onto the +/-1 convention LIBERO consumes.  Working post-mapping keeps
"freeze" and "mirror" unambiguous: dimension 6 is already a signed gripper
command, and dimensions 0-5 are already the deltas the controller integrates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Action layout of a LIBERO OSC command: (dx, dy, dz, drx, dry, drz, gripper).
XY_SLICE = slice(0, 2)
DELTA_SLICE = slice(0, 6)
GRIPPER_INDEX = 6

SELF = "SELF"
HIJACK = "HIJACK"


def apply_mirror(chunk: np.ndarray) -> np.ndarray:
    """T1: negate the x/y translation deltas, keep z, rotation and gripper."""
    out = np.array(chunk, copy=True)
    out[..., XY_SLICE] *= -1.0
    return out


def apply_freeze(chunk: np.ndarray) -> np.ndarray:
    """T2: zero every delta, hold the commanded gripper."""
    out = np.array(chunk, copy=True)
    out[..., DELTA_SLICE] = 0.0
    return out


def apply_swap(chunk: np.ndarray, donor_chunk: np.ndarray) -> np.ndarray:
    """T3: execute another env's commanded chunk verbatim."""
    return np.array(donor_chunk, copy=True)


@dataclass
class HijackDecision:
    """What happened to one env at one model call."""

    env_index: int
    hijacked: bool
    transform: Optional[str] = None
    donor_env_index: Optional[int] = None
    # Set when a hijack was scheduled but could not be realised (e.g. a swap
    # with no eligible donor).  Kept so the logbook can show it was not silent.
    skipped_reason: Optional[str] = None


class HijackScheduler:
    """Per-env i.i.d. hijack schedule with a mandatory clean gap.

    A call is eligible when the episode is past warm-up, still active (not yet
    successful), and at least ``min_clean_gap`` clean calls have elapsed since
    the previous hijack.  Enforcing the gap is what makes the positive sample
    ``h_{m+1}`` reflect exactly one manipulated transition.
    """

    def __init__(
        self,
        num_envs: int,
        p_hijack: float,
        warmup_calls: int,
        min_clean_gap: int,
        transform: str,
        rng: np.random.Generator,
    ) -> None:
        if min_clean_gap < 1:
            raise ValueError("min_clean_gap must be >= 1")
        self.num_envs = num_envs
        self.p_hijack = p_hijack
        self.warmup_calls = warmup_calls
        self.min_clean_gap = min_clean_gap
        self.transform = transform
        self.rng = rng
        # -inf sentinel so the first eligible call is never gap-blocked.
        self._last_hijack_call = np.full(num_envs, -(10**9), dtype=np.int64)

    def reset(self, env_indices: Optional[np.ndarray] = None) -> None:
        """Clear hijack history, for a whole batch or selected env slots."""
        if env_indices is None:
            self._last_hijack_call[:] = -(10**9)
        else:
            self._last_hijack_call[np.asarray(env_indices, dtype=int)] = -(10**9)

    def eligible(self, call_idx: int, active: np.ndarray) -> np.ndarray:
        """Boolean mask of envs allowed to be hijacked at ``call_idx``."""
        active = np.asarray(active, dtype=bool)
        past_warmup = call_idx >= self.warmup_calls
        gap_ok = (call_idx - self._last_hijack_call) > self.min_clean_gap
        return active & gap_ok & past_warmup

    def decide(
        self,
        call_idx: int,
        active: np.ndarray,
        donor_active: Optional[np.ndarray] = None,
    ) -> list[HijackDecision]:
        """Draw hijack decisions for one model call.

        Args:
            call_idx: index of the model call being decided.
            active: envs eligible to *be* hijacked.
            donor_active: envs eligible to *donate* a chunk for a swap.
                Defaults to ``active``, but the two differ in practice: an
                episode excluded from hijacking (a clean control episode) still
                produces perfectly ordinary policy output and makes a valid
                donor.  Keeping the sets separate is what stops the donor pool
                from silently shrinking.
        """
        active = np.asarray(active, dtype=bool)
        donors = (
            active if donor_active is None else np.asarray(donor_active, dtype=bool)
        )
        eligible = self.eligible(call_idx, active)
        draws = self.rng.random(self.num_envs) < self.p_hijack
        active_indices = np.flatnonzero(donors)

        decisions: list[HijackDecision] = []
        for env_index in range(self.num_envs):
            if not (eligible[env_index] and draws[env_index]):
                decisions.append(HijackDecision(env_index=env_index, hijacked=False))
                continue

            donor_index = None
            if self.transform == "swap":
                candidates = active_indices[active_indices != env_index]
                if candidates.size == 0:
                    decisions.append(
                        HijackDecision(
                            env_index=env_index,
                            hijacked=False,
                            skipped_reason="no_swap_donor",
                        )
                    )
                    continue
                donor_index = int(self.rng.choice(candidates))

            self._last_hijack_call[env_index] = call_idx
            decisions.append(
                HijackDecision(
                    env_index=env_index,
                    hijacked=True,
                    transform=self.transform,
                    donor_env_index=donor_index,
                )
            )
        return decisions


def build_executed_chunk(
    commanded: np.ndarray, decisions: list[HijackDecision]
) -> np.ndarray:
    """Apply a batch of decisions to ``commanded`` ``[num_envs, chunk, dim]``.

    Donor chunks are always read from ``commanded`` (the pre-hijack commands),
    so a chain of swaps cannot propagate an already-hijacked chunk.
    """
    commanded = np.asarray(commanded)
    if commanded.ndim != 3:
        raise ValueError(f"expected [num_envs, chunk, dim], got {commanded.shape}")
    executed = np.array(commanded, copy=True)
    for decision in decisions:
        if not decision.hijacked:
            continue
        env_index = decision.env_index
        if decision.transform == "mirror":
            executed[env_index] = apply_mirror(commanded[env_index])
        elif decision.transform == "freeze":
            executed[env_index] = apply_freeze(commanded[env_index])
        elif decision.transform == "swap":
            executed[env_index] = apply_swap(
                commanded[env_index], commanded[decision.donor_env_index]
            )
        else:
            raise ValueError(f"unknown transform: {decision.transform!r}")
    return executed
