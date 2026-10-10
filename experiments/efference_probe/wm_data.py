"""World-model queries built from a collection run (wave 3, E3/E4).

Torch-free: reads ``calls.parquet`` and the saved frames, and assembles, for a
model call ``m``, everything the Wan world model conditions on plus the action
chunks to try.  The conditioning follows ``rlinf/envs/world_model/
world_model_wan_env.py`` (``_infer_next_chunk_frames``):

* **reference** -- the episode's first frame (``call0000_main.jpg``); the WM
  keeps it fixed for the whole episode.
* **context** -- the 4 most recent per-step frames.
* **condition actions** -- 5 slots: slot 0 is the reference frame's action
  (zeros, gripper -1, as at reset), slots 1-4 the actions that produced the
  4 context frames.
* **chunk** -- the 8 x 7 action chunk in LIBERO's executable convention
  (OFT deltas, gripper +/-1), i.e. ``a_cmd_env``/``a_exec`` as logged.

Two context modes:

``substeps``  faithful.  Needs a run collected with ``frames.substeps``: the
              context is the last 4 substep frames of call ``m-1``'s chunk,
              the condition actions are the last 4 rows of ``a_exec[m-1]``,
              and the real outcome is the 8 substep frames of call ``m``.
``static``    an approximation for runs with one frame per call (waves 1-2):
              the context is call ``m``'s frame repeated 4 times with "held
              still" condition actions (zero deltas, gripper as last executed),
              and the real outcome is call ``m+1``'s frame only.  This is the
              WM's own reset convention applied mid-episode -- usable for the
              E3 sensitivity gate on existing data, off-distribution for E4.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from .datasets import RunData
from .hijack import apply_freeze, apply_mirror
from .storage import call_frame_name, substep_frame_name

CONTEXT_MODES = ("substeps", "static")
N_CONTEXT = 4
N_CONDITION = 5
GRIPPER_OPEN = -1.0


@dataclass
class WMQuery:
    """One world-model question about model call ``call_idx``."""

    episode_id: int
    call_idx: int
    task_id: int
    label: str
    transform: str
    reference: np.ndarray  # [H, W, 3] uint8
    context: np.ndarray  # [4, H, W, 3] uint8
    condition: np.ndarray  # [5, 7] float32
    candidates: "OrderedDict[str, np.ndarray]"  # name -> [chunk, 7] float32
    actual: Optional[np.ndarray]  # [n_frames, H, W, 3] uint8, or None
    context_mode: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def alternatives(self) -> list[str]:
        return [name for name in self.candidates if name.startswith("alt")]


class FrameStore:
    """Lazy JPEG reader with a small LRU cache."""

    def __init__(self, run_dir: str | Path, max_items: int = 512) -> None:
        self.frames_dir = Path(run_dir) / "frames"
        self.max_items = max_items
        self._cache: "OrderedDict[Path, np.ndarray]" = OrderedDict()

    def _read(self, path: Path) -> np.ndarray:
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached
        if not path.is_file():
            raise FileNotFoundError(f"missing frame {path}")
        from PIL import Image

        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        self._cache[path] = array
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return array

    def call_frame(self, episode_id: int, call_idx: int) -> np.ndarray:
        return self._read(
            self.frames_dir / f"ep{episode_id:05d}" / call_frame_name(call_idx)
        )

    def substep_frame(self, episode_id: int, call_idx: int, substep: int) -> np.ndarray:
        return self._read(
            self.frames_dir
            / f"ep{episode_id:05d}"
            / substep_frame_name(call_idx, substep)
        )

    def has_call_frame(self, episode_id: int, call_idx: int) -> bool:
        return (
            self.frames_dir / f"ep{episode_id:05d}" / call_frame_name(call_idx)
        ).is_file()

    def has_substeps(self, episode_id: int, call_idx: int) -> bool:
        return (
            self.frames_dir / f"ep{episode_id:05d}" / substep_frame_name(call_idx, 0)
        ).is_file()


def condition_actions(prev_executed: Optional[np.ndarray], mode: str) -> np.ndarray:
    """The 5 conditioning action slots for the WM (see module docstring)."""
    if mode not in CONTEXT_MODES:
        raise ValueError(f"unknown context mode {mode!r}; use {CONTEXT_MODES}")
    condition = np.zeros((N_CONDITION, 7), dtype=np.float32)
    condition[:, 6] = GRIPPER_OPEN
    if prev_executed is None:
        return condition  # call 0: the WM's own reset convention
    prev_executed = np.asarray(prev_executed, dtype=np.float32)
    if mode == "substeps":
        condition[1:] = prev_executed[-(N_CONDITION - 1) :]
    else:
        # Held still for 4 steps, gripper where the last chunk left it.
        condition[1:, 6] = prev_executed[-1, 6]
    return condition


class ChunkTable:
    """Chunks and labels of a run, addressable by ``(episode_id, call_idx)``."""

    def __init__(self, run: RunData) -> None:
        calls = run.calls
        n_chunks = int(run.schema["num_action_chunks"])
        action_dim = int(run.schema["action_dim"])
        self.shape = (n_chunks, action_dim)
        self.calls = calls.reset_index(drop=True)
        self.position = {
            (int(e), int(c)): i
            for i, (e, c) in enumerate(zip(calls["episode_id"], calls["call_idx"]))
        }
        self._cmd = calls["a_cmd_env"].to_numpy()
        self._exec = calls["a_exec"].to_numpy()
        self._by_task: dict[int, pd.DataFrame] = {
            int(task): group[["episode_id", "call_idx"]].reset_index()
            for task, group in self.calls.groupby("task_id")
        }

    def commanded(self, episode_id: int, call_idx: int) -> np.ndarray:
        row = self.position[(int(episode_id), int(call_idx))]
        return np.asarray(self._cmd[row], dtype=np.float32).reshape(self.shape)

    def executed(self, episode_id: int, call_idx: int) -> np.ndarray:
        row = self.position[(int(episode_id), int(call_idx))]
        return np.asarray(self._exec[row], dtype=np.float32).reshape(self.shape)

    def row(self, episode_id: int, call_idx: int) -> pd.Series:
        return self.calls.iloc[self.position[(int(episode_id), int(call_idx))]]

    def donors(
        self,
        task_id: int,
        episode_id: int,
        call_idx: int,
        count: int,
        rng: np.random.Generator,
        exclude_episodes: Iterable[int] = (),
    ) -> list[tuple[int, int]]:
        """``count`` other same-task episodes' calls nearest in phase.

        The swap-donor distribution: other initial states of the same task at
        (as near as possible) the same call index, one call per episode.
        Ties within a phase-distance tier are broken at random.
        """
        pool = self._by_task.get(int(task_id))
        if pool is None:
            return []
        excluded = {int(episode_id), *(int(e) for e in exclude_episodes)}
        pool = pool[~pool["episode_id"].isin(excluded)]
        if pool.empty:
            return []
        distance = (pool["call_idx"] - int(call_idx)).abs().to_numpy()
        jitter = rng.random(len(pool))
        order = np.lexsort((jitter, distance))
        chosen: list[tuple[int, int]] = []
        seen: set[int] = set()
        for index in order:
            episode = int(pool["episode_id"].iloc[index])
            if episode in seen:
                continue
            seen.add(episode)
            chosen.append((episode, int(pool["call_idx"].iloc[index])))
            if len(chosen) == count:
                break
        return chosen


def build_query(
    table: ChunkTable,
    frames: FrameStore,
    episode_id: int,
    call_idx: int,
    mode: str,
    n_alternatives: int,
    rng: np.random.Generator,
    with_transforms: bool = False,
) -> WMQuery:
    """Assemble the WM question for one call.

    Candidates, in order: ``cmd`` (what the policy commanded), ``exec`` (what
    was executed, only when it differs), optional ``mirror``/``freeze``/``swap``
    transforms of ``cmd`` (E3), then ``alt1..altK`` -- other same-task
    episodes' commanded chunks at matched phase, which define the
    action-marginal prediction.  A logged swap donor is excluded from the
    alternatives, so the executed chunk is never one of them.
    """
    if mode not in CONTEXT_MODES:
        raise ValueError(f"unknown context mode {mode!r}; use {CONTEXT_MODES}")
    row = table.row(episode_id, call_idx)
    task_id = int(row["task_id"])
    reference = frames.call_frame(episode_id, 0)
    has_prev = call_idx > 0 and (episode_id, call_idx - 1) in table.position
    prev_executed = table.executed(episode_id, call_idx - 1) if has_prev else None

    if mode == "substeps":
        if has_prev:
            context = np.stack(
                [frames.substep_frame(episode_id, call_idx - 1, k) for k in range(4, 8)]
            )
        else:
            context = np.stack([reference] * N_CONTEXT)
        actual = np.stack(
            [
                frames.substep_frame(episode_id, call_idx, k)
                for k in range(table.shape[0])
            ]
        )
    else:
        context = np.stack([frames.call_frame(episode_id, call_idx)] * N_CONTEXT)
        actual = (
            frames.call_frame(episode_id, call_idx + 1)[None]
            if frames.has_call_frame(episode_id, call_idx + 1)
            else None
        )

    commanded = table.commanded(episode_id, call_idx)
    executed = table.executed(episode_id, call_idx)
    candidates: "OrderedDict[str, np.ndarray]" = OrderedDict(cmd=commanded)
    if not np.allclose(executed, commanded):
        candidates["exec"] = executed

    exclude = []
    donor_episode = int(row.get("donor_episode_id", -1))
    if donor_episode >= 0:
        exclude.append(donor_episode)
    donors = table.donors(
        task_id,
        episode_id,
        call_idx,
        n_alternatives + (1 if with_transforms else 0),
        rng,
        exclude_episodes=exclude,
    )
    if with_transforms:
        candidates["mirror"] = apply_mirror(commanded)
        candidates["freeze"] = apply_freeze(commanded)
        if donors:
            candidates["swap"] = table.commanded(*donors[0])
            donors = donors[1:]
    for k, donor in enumerate(donors, start=1):
        candidates[f"alt{k}"] = table.commanded(*donor)

    return WMQuery(
        episode_id=int(episode_id),
        call_idx=int(call_idx),
        task_id=task_id,
        label=str(row["label"]),
        transform=str(row["transform"]),
        reference=reference,
        context=context,
        condition=condition_actions(prev_executed, mode),
        candidates=candidates,
        actual=actual,
        context_mode=mode,
        meta={"donors": donors, "swap_donor_episode": donor_episode},
    )


def detect_context_mode(run: RunData) -> str:
    """``substeps`` when the run saved them, else ``static``."""
    frames_cfg = (run.config.get("config") or {}).get("frames") or {}
    return "substeps" if frames_cfg.get("substeps") else "static"


def calls_with_frames(run: RunData, frames: FrameStore, mode: str) -> pd.DataFrame:
    """Calls whose query (context, outcome, reference) is fully on disk."""
    calls = run.calls
    keep = []
    for episode_id, call_idx in zip(calls["episode_id"], calls["call_idx"]):
        episode_id, call_idx = int(episode_id), int(call_idx)
        if not frames.has_call_frame(episode_id, 0):
            keep.append(False)
            continue
        if mode == "substeps":
            ok = frames.has_substeps(episode_id, call_idx) and (
                call_idx == 0 or frames.has_substeps(episode_id, call_idx - 1)
            )
        else:
            ok = frames.has_call_frame(episode_id, call_idx) and frames.has_call_frame(
                episode_id, call_idx + 1
            )
        keep.append(ok)
    return calls[np.asarray(keep, dtype=bool)]


__all__ = [
    "CONTEXT_MODES",
    "ChunkTable",
    "FrameStore",
    "WMQuery",
    "build_query",
    "calls_with_frames",
    "condition_actions",
    "detect_context_mode",
]
