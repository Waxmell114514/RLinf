"""X4: the external comparator and its recovery action (pure numpy).

PREREG_X1_X4.md, X4.  The monitor is P2r's comparison block on the *true*
commanded chunk and the *true* state change, scored by a frozen linear model
(the ladder's StandardScaler + logistic regression, exported as numbers so the
harness needs no scikit-learn).  When it fires on call c, call c+1 executes the
undo of chunk c instead of the policy's output.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .datasets import _axis_angle_to_matrix, _matrix_to_axis_angle

CONDITIONS = ("off", "monitor", "oracle")
# Seeds the per-episode condition draw; separate from the hijack stream
# (0xEFFE) so assigning conditions never changes the hijack schedule.
CONDITION_STREAM = 0x4D4F4E
RECOVERY = "RECOVERY"
DELTA_DIMS = 6  # dx, dy, dz, drx, dry, drz; dimension 6 is the gripper


def delta_state(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """``datasets._delta_states`` on arrays, at the parquet's float32 precision."""
    before = np.asarray(before, dtype=np.float32)
    after = np.asarray(after, dtype=np.float32)
    delta = after - before
    if after.shape[1] >= 6:
        relative = _axis_angle_to_matrix(
            after[:, 3:6].astype(np.float64)
        ) @ np.transpose(
            _axis_angle_to_matrix(before[:, 3:6].astype(np.float64)), (0, 2, 1)
        )
        delta[:, 3:6] = _matrix_to_axis_angle(relative).astype(delta.dtype)
    return delta


def mismatch_features(
    commanded: np.ndarray, before: np.ndarray, after: np.ndarray
) -> np.ndarray:
    """P2r's comparison block, identical to ``datasets._mismatch_features``.

    Args:
        commanded: ``[n, num_action_chunks, action_dim]`` env-space command.
        before, after: ``[n, state_dim]`` states around the chunk.
    """
    commanded = np.asarray(commanded, dtype=np.float32)
    summed = commanded.sum(axis=1)
    delta = delta_state(before, after)
    n_shared = min(summed.shape[1] - 1, delta.shape[1])
    parts = [summed, delta, summed[:, :n_shared] * delta[:, :n_shared]]
    for lo, hi in ((0, 3), (3, 6)):
        if summed.shape[1] < hi or delta.shape[1] < hi:
            continue
        command_block, delta_block = summed[:, lo:hi], delta[:, lo:hi]
        command_norm = np.linalg.norm(command_block, axis=1)
        delta_norm = np.linalg.norm(delta_block, axis=1)
        denominator = np.maximum(command_norm * delta_norm, 1e-8)
        cosine = (command_block * delta_block).sum(axis=1) / denominator
        parts.append(
            np.stack(
                [
                    command_norm,
                    delta_norm,
                    cosine,
                    delta_norm - command_norm,
                    np.linalg.norm(delta_block - command_block, axis=1),
                ],
                axis=1,
            )
        )
    return np.concatenate(parts, axis=1).astype(np.float32)


def undo_chunk(executed: np.ndarray) -> np.ndarray:
    """Retrace an executed chunk: steps reversed, deltas negated, gripper kept.

    ``undo[k] = (-executed[n-1-k][0:6], executed[n-1-k][6])``.  Every step
    stays inside the action bounds because the executed chunk was.
    """
    undo = np.array(executed, dtype=np.float32, copy=True)[::-1].copy()
    undo[:, :DELTA_DIMS] *= -1.0
    return undo


def assign_conditions(
    seed: int, task_id: int, batch_index: int, n: int, probs: list[float]
) -> np.ndarray:
    """One condition per episode slot, from the dedicated condition stream."""
    rng = np.random.default_rng([seed, task_id, batch_index, CONDITION_STREAM])
    picks = rng.choice(len(CONDITIONS), size=n, p=np.asarray(probs, dtype=np.float64))
    return np.asarray([CONDITIONS[i] for i in picks], dtype=object)


@dataclass
class LinearMonitor:
    """A frozen StandardScaler + logistic regression, as plain numbers."""

    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    threshold: float
    meta: dict[str, Any] = field(default_factory=dict)
    # The artifact's digest, set by ``load``; not part of the payload.
    sha256: str = ""

    def score(self, features: np.ndarray) -> np.ndarray:
        """The pipeline's ``decision_function``: larger means more hijack-like."""
        x = np.asarray(features, dtype=np.float64)
        if x.shape[-1] != self.coef.shape[0]:
            raise ValueError(
                f"monitor expects {self.coef.shape[0]} features, got {x.shape[-1]}"
            )
        return ((x - self.mean) / self.scale) @ self.coef + self.intercept

    def fires(self, features: np.ndarray) -> np.ndarray:
        return self.score(features) > self.threshold

    # -- the frozen artifact -------------------------------------------------

    def payload(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": float(self.intercept),
            "threshold": float(self.threshold),
            "meta": self.meta,
        }

    @staticmethod
    def digest(payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    def save(self, path: str | Path) -> str:
        payload = self.payload()
        digest = self.digest(payload)
        Path(path).write_text(
            json.dumps({"monitor": payload, "sha256": digest}, indent=2),
            encoding="utf-8",
        )
        return digest

    @classmethod
    def load(cls, path: str | Path, expect_sha256: Optional[str] = None):
        stored = json.loads(Path(path).read_text(encoding="utf-8"))
        payload, digest = stored["monitor"], stored["sha256"]
        if cls.digest(payload) != digest:
            raise ValueError(f"{path}: payload does not match its recorded sha256")
        if expect_sha256 and digest != expect_sha256:
            raise ValueError(f"{path}: sha256 {digest} != expected {expect_sha256}")
        monitor = cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            coef=np.asarray(payload["coef"], dtype=np.float64),
            intercept=float(payload["intercept"]),
            threshold=float(payload["threshold"]),
            meta=payload.get("meta", {}),
            sha256=digest,
        )
        return monitor
