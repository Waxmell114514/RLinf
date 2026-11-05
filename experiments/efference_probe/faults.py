"""X6: persistent actuation faults on the executed chunk.

Pre-registered in ``logbook.md`` ("X6 STAGE-0 PRE-REGISTRATION").  A fault is
a pure function of the chunk the env is about to execute, in env action space;
it applies on every call of an episode.  Translation/rotation dims (0-5) are
clipped back to the action bounds afterwards; the gripper (dim 6) is untouched.

Specs: ``clean``, ``invert_x``, ``invert_y``, ``bias_x:<b>`` (b per step on x),
``bias:<m>@<deg>`` (a bias of norm m in the horizontal plane, at angle deg from
+x), ``gain:<g>`` (g on translation dims 0-2), ``rotate:<deg>`` (the
horizontal (x, y) command rotated by deg: a misaligned action frame).

X6-EC (``PREREG_X6.md`` §3.2) adds the oracle's inverse (:meth:`Fault.compensate`)
and the per-episode draw of the training collection (:func:`draw_training_fault`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FAMILIES = ("clean", "invert_x", "invert_y", "bias_x", "bias", "gain", "rotate")
BOUND = 1.0


@dataclass(frozen=True)
class Fault:
    spec: str
    family: str
    magnitude: float = 0.0
    angle_deg: float = 0.0

    @property
    def bias_vector(self) -> np.ndarray:
        """The per-step (x, y) bias of a bias fault."""
        if self.family == "bias_x":
            return np.array([self.magnitude, 0.0], np.float32)
        theta = np.deg2rad(self.angle_deg)
        return np.array(
            [self.magnitude * np.cos(theta), self.magnitude * np.sin(theta)],
            np.float32,
        )

    @property
    def dims(self) -> tuple[int, ...]:
        """The action dims the fault changes."""
        return {
            "clean": (),
            "invert_x": (0,),
            "invert_y": (1,),
            "bias_x": (0,),
            "bias": (0, 1),
            "gain": (0, 1, 2),
            "rotate": (0, 1),
        }[self.family]

    @property
    def rotation(self) -> np.ndarray:
        """The 2 x 2 rotation a rotate fault applies to (x, y)."""
        theta = np.deg2rad(self.magnitude)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]], np.float32)

    def apply(self, executed: np.ndarray) -> np.ndarray:
        """The faulted copy of ``executed`` [..., chunks, dim]."""
        out = np.array(executed, dtype=np.float32, copy=True)
        if self.family == "clean":
            return out
        if self.family == "invert_x":
            out[..., 0] *= -1.0
        elif self.family == "invert_y":
            out[..., 1] *= -1.0
        elif self.family in ("bias_x", "bias"):
            out[..., 0:2] += self.bias_vector
        elif self.family == "gain":
            out[..., 0:3] *= self.magnitude
        elif self.family == "rotate":
            out[..., 0:2] = out[..., 0:2] @ self.rotation.T
        out[..., :6] = np.clip(out[..., :6], -BOUND, BOUND)
        return out

    def compensate(self, command: np.ndarray) -> np.ndarray:
        """The oracle's command: what to send so the fault executes ``command``.

        Exact unless the result leaves the action bounds, where it is clipped
        (so a large command under a large bias is compensated only in part).
        """
        out = np.array(command, dtype=np.float32, copy=True)
        if self.family == "clean":
            return out
        if self.family == "invert_x":
            out[..., 0] *= -1.0
        elif self.family == "invert_y":
            out[..., 1] *= -1.0
        elif self.family in ("bias_x", "bias"):
            out[..., 0:2] -= self.bias_vector
        elif self.family == "gain":
            out[..., 0:3] /= self.magnitude
        elif self.family == "rotate":
            out[..., 0:2] = out[..., 0:2] @ self.rotation
        out[..., :6] = np.clip(out[..., :6], -BOUND, BOUND)
        return out


def parse(spec: str) -> Fault:
    family, _, value = spec.partition(":")
    if family not in FAMILIES:
        raise ValueError(f"unknown fault family in {spec!r}; expected {FAMILIES}")
    if family in ("clean", "invert_x", "invert_y"):
        if value:
            raise ValueError(f"{family} takes no magnitude: {spec!r}")
        return Fault(spec, family)
    if not value:
        raise ValueError(f"{family} needs a magnitude, e.g. {family}:0.3")
    if family == "bias":
        norm, at, angle = value.partition("@")
        if not at:
            raise ValueError(f"bias needs a direction, e.g. bias:0.3@45: {spec!r}")
        magnitude = float(norm)
        if not 0 < magnitude < BOUND:
            raise ValueError(f"bias magnitude must be in (0, 1): {spec!r}")
        return Fault(spec, family, magnitude, float(angle) % 360.0)
    magnitude = float(value)
    if family == "bias_x" and not 0 < abs(magnitude) < BOUND:
        raise ValueError(f"bias magnitude must be in (0, 1): {spec!r}")
    if family == "gain" and not 0 < magnitude <= 1.5:
        raise ValueError(f"gain must be in (0, 1.5]: {spec!r}")
    if family == "rotate" and not 0 < abs(magnitude) <= 180:
        raise ValueError(f"rotation must be in (0, 180] degrees: {spec!r}")
    return Fault(spec, family, magnitude)


# -- X6-EC training collection (PREREG_X6.md section 3.2) ------------------------

P_CLEAN, P_BIAS, P_INVERT = 0.30, 0.45, 0.25
BIAS_RANGE = (0.10, 0.35)
# Bias directions within this many degrees of an axis are never drawn: the
# axis directions are the held-out evaluation directions.
HELD_OUT_HALF_WIDTH_DEG = 10.0
P_COMPENSATED = 0.70


@dataclass(frozen=True)
class FaultDraw:
    """One training episode's fault, and whether the oracle compensates it."""

    fault: Fault
    compensated: bool


def draw_training_fault(rng: np.random.Generator) -> FaultDraw:
    """The per-episode draw of the X6-EC training collection."""
    u = rng.random()
    if u < P_CLEAN:
        return FaultDraw(parse("clean"), False)
    if u < P_CLEAN + P_BIAS:
        magnitude = rng.uniform(*BIAS_RANGE)
        # Four allowed arcs, each between two held-out axis sectors.
        arc = 90.0 - 2 * HELD_OUT_HALF_WIDTH_DEG
        angle = 90.0 * rng.integers(4) + HELD_OUT_HALF_WIDTH_DEG + arc * rng.random()
        fault = parse(f"bias:{magnitude:.4f}@{angle:.2f}")
    else:
        fault = parse("invert_x" if rng.random() < 0.5 else "invert_y")
    return FaultDraw(fault, bool(rng.random() < P_COMPENSATED))
