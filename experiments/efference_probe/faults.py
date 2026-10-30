"""X6: persistent actuation faults on the executed chunk.

Pre-registered in ``logbook.md`` ("X6 STAGE-0 PRE-REGISTRATION").  A fault is
a pure function of the chunk the env is about to execute, in env action space;
it applies on every call of an episode.  Translation/rotation dims (0-5) are
clipped back to the action bounds afterwards; the gripper (dim 6) is untouched.

Specs: ``clean``, ``invert_x``, ``bias_x:<b>`` (b per step on x), ``gain:<g>``
(g on translation dims 0-2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FAMILIES = ("clean", "invert_x", "bias_x", "gain")
BOUND = 1.0


@dataclass(frozen=True)
class Fault:
    spec: str
    family: str
    magnitude: float = 0.0

    def apply(self, executed: np.ndarray) -> np.ndarray:
        """The faulted copy of ``executed`` [..., chunks, dim]."""
        out = np.array(executed, dtype=np.float32, copy=True)
        if self.family == "clean":
            return out
        if self.family == "invert_x":
            out[..., 0] *= -1.0
        elif self.family == "bias_x":
            out[..., 0] += self.magnitude
        elif self.family == "gain":
            out[..., 0:3] *= self.magnitude
        out[..., :6] = np.clip(out[..., :6], -BOUND, BOUND)
        return out


def parse(spec: str) -> Fault:
    family, _, value = spec.partition(":")
    if family not in FAMILIES:
        raise ValueError(f"unknown fault family in {spec!r}; expected {FAMILIES}")
    if family in ("clean", "invert_x"):
        if value:
            raise ValueError(f"{family} takes no magnitude: {spec!r}")
        return Fault(spec, family)
    if not value:
        raise ValueError(f"{family} needs a magnitude, e.g. {family}:0.3")
    magnitude = float(value)
    if family == "bias_x" and not 0 < abs(magnitude) < BOUND:
        raise ValueError(f"bias magnitude must be in (0, 1): {spec!r}")
    if family == "gain" and not 0 < magnitude <= 1.5:
        raise ValueError(f"gain must be in (0, 1.5]: {spec!r}")
    return Fault(spec, family, magnitude)
