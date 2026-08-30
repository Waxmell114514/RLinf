"""Run-directory layout and writers.

Depends only on numpy/pandas/pyarrow (plus Pillow for frames), so the analysis
half of the project can read a run without torch installed.

Layout under ``<out_root>/<run_id>/``::

    run_config.yaml     resolved config, git SHA, checkpoint, verification report
    calls.parquet       one row per model call
    hidden/ep00000.npz  per-episode hidden states, float16
    frames/ep00000/     main/wrist jpegs
    manifest.json       counts, budget usage, per-episode summary

Array columns in ``calls.parquet`` are stored flattened; their shapes are
recorded in ``run_config.yaml`` under ``schema`` and re-applied by
:func:`unflatten`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

# Flattened array columns -> shape, so analysis code never guesses.
ARRAY_COLUMNS: dict[str, tuple[str, ...]] = {
    "a_cmd_model": ("num_action_chunks", "action_dim"),
    "a_cmd_env": ("num_action_chunks", "action_dim"),
    "a_exec": ("num_action_chunks", "action_dim"),
    "states_before": ("state_dim",),
    "states_after": ("state_dim",),
    "logprobs": ("n_act",),
    "entropy_tokens": ("n_act",),
}


def flatten(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32).reshape(-1)


def unflatten(frame: pd.DataFrame, column: str, shape: tuple[int, ...]) -> np.ndarray:
    """Stack a flattened list column back into ``[n_rows, *shape]``."""
    stacked = np.stack([np.asarray(v, dtype=np.float32) for v in frame[column]])
    return stacked.reshape(len(frame), *shape)


class DiskBudget:
    """Tracks bytes written and wall-clock, so a run stops before it sprawls."""

    def __init__(self, root: Path, max_gb: float, max_hours: float) -> None:
        self.root = Path(root)
        self.max_bytes = max_gb * (1024**3)
        self.max_seconds = max_hours * 3600.0
        self.start = time.time()
        self._bytes = 0
        self._output_bytes = 0

    def add_bytes(self, count: int) -> None:
        self._bytes += int(count)

    def set_output_bytes(self, count: int) -> None:
        """Size of files that get rewritten in place, tracked separately."""
        self._output_bytes = int(count)

    @property
    def bytes_written(self) -> int:
        return self._bytes + self._output_bytes

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start

    def exceeded(self) -> Optional[str]:
        """Return the name of the exhausted budget, or None."""
        if self.bytes_written >= self.max_bytes:
            return "disk"
        if self.elapsed_seconds >= self.max_seconds:
            return "time"
        return None

    def summary(self) -> dict[str, float]:
        return {
            "disk_gb": self.bytes_written / (1024**3),
            "elapsed_hours": self.elapsed_seconds / 3600.0,
            "max_disk_gb": self.max_bytes / (1024**3),
            "max_hours": self.max_seconds / 3600.0,
        }


class RunWriter:
    """Accumulates call rows and per-episode hidden states, then flushes."""

    def __init__(self, out_dir: str | Path, budget: DiskBudget) -> None:
        self.out_dir = Path(out_dir)
        self.hidden_dir = self.out_dir / "hidden"
        self.frames_dir = self.out_dir / "frames"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.hidden_dir.mkdir(exist_ok=True)
        self.budget = budget

        self._rows: list[dict[str, Any]] = []
        self._hidden: dict[int, list[np.ndarray]] = {}
        self._vision: dict[int, list[np.ndarray]] = {}
        self._hidden_calls: dict[int, list[int]] = {}
        self._episodes: list[dict[str, Any]] = []

    # -- rows ------------------------------------------------------------

    def add_call(self, row: dict[str, Any]) -> None:
        self._rows.append(row)

    def add_episode(self, summary: dict[str, Any]) -> None:
        self._episodes.append(summary)

    @property
    def n_calls(self) -> int:
        return len(self._rows)

    # -- hidden states ---------------------------------------------------

    def add_hidden(
        self,
        episode_id: int,
        call_idx: int,
        hidden: np.ndarray,
        vision: Optional[np.ndarray] = None,
    ) -> None:
        """Buffer one call's pooled hidden states ``[n_layers, n_pools, D]``."""
        self._hidden.setdefault(episode_id, []).append(
            np.asarray(hidden, dtype=np.float16)
        )
        self._hidden_calls.setdefault(episode_id, []).append(int(call_idx))
        if vision is not None:
            self._vision.setdefault(episode_id, []).append(
                np.asarray(vision, dtype=np.float16)
            )

    def flush_episode(self, episode_id: int) -> None:
        """Write and drop one episode's buffered hidden states."""
        calls = self._hidden.pop(episode_id, None)
        call_indices = self._hidden_calls.pop(episode_id, [])
        vision = self._vision.pop(episode_id, None)
        if not calls:
            return
        payload = {
            "h": np.stack(calls),  # [n_calls, n_layers, n_pools, D]
            "call_idx": np.asarray(call_indices, dtype=np.int32),
        }
        if vision:
            payload["vis"] = np.stack(vision)
        path = self.hidden_dir / f"ep{episode_id:05d}.npz"
        np.savez_compressed(path, **payload)
        self.budget.add_bytes(path.stat().st_size)

    # -- frames ----------------------------------------------------------

    def save_frames(
        self,
        episode_id: int,
        call_idx: int,
        main_image: np.ndarray,
        wrist_image: Optional[np.ndarray] = None,
        quality: int = 85,
    ) -> None:
        from PIL import Image

        directory = self.frames_dir / f"ep{episode_id:05d}"
        directory.mkdir(parents=True, exist_ok=True)
        images = {"main": main_image}
        if wrist_image is not None:
            images["wrist"] = wrist_image
        for name, array in images.items():
            path = directory / f"call{call_idx:04d}_{name}.jpg"
            Image.fromarray(np.asarray(array, dtype=np.uint8)).save(
                path, quality=quality
            )
            self.budget.add_bytes(path.stat().st_size)

    # -- finalisation ----------------------------------------------------

    def write_run_config(self, payload: dict[str, Any]) -> None:
        path = self.out_dir / "run_config.yaml"
        with open(path, "w") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)

    def checkpoint(self, manifest_extra: Optional[dict[str, Any]] = None) -> None:
        """Write calls.parquet and the manifest with everything buffered so far.

        Call rows live in memory until they are written, so a crash three hours
        into a run would otherwise leave a directory full of hidden-state
        archives with no labels to go with them -- the hidden states alone are
        unusable, and the GPU hours unrecoverable.  Rewriting the whole parquet
        after each batch is a few milliseconds at this scale.
        """
        self._write(manifest_extra)

    def finalize(self, manifest_extra: Optional[dict[str, Any]] = None) -> Path:
        """Flush any remaining episodes and write calls.parquet + manifest."""
        for episode_id in list(self._hidden):
            self.flush_episode(episode_id)
        return self._write(manifest_extra)

    def _write(self, manifest_extra: Optional[dict[str, Any]] = None) -> Path:
        frame = pd.DataFrame(self._rows)
        parquet_path = self.out_dir / "calls.parquet"
        if not frame.empty:
            frame.to_parquet(parquet_path, index=False)

        manifest = {
            "n_calls": len(self._rows),
            "n_episodes": len(self._episodes),
            "budget": self.budget.summary(),
            "episodes": self._episodes,
        }
        if manifest_extra:
            manifest.update(manifest_extra)
        with open(self.out_dir / "manifest.json", "w") as handle:
            json.dump(manifest, handle, indent=2, default=_json_default)
        # Budget accounting reads the directory rather than accumulating
        # per-write sizes, so repeated checkpoints do not double-count.
        self.budget.set_output_bytes(
            sum(
                path.stat().st_size
                for path in (parquet_path, self.out_dir / "manifest.json")
                if path.exists()
            )
        )
        return parquet_path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def git_sha(repo_root: str | Path) -> str:
    """Best-effort ``git rev-parse HEAD`` without shelling out to git."""
    head = Path(repo_root) / ".git" / "HEAD"
    try:
        content = head.read_text().strip()
    except OSError:
        return "unknown"
    if content.startswith("ref:"):
        ref = content.split(" ", 1)[1].strip()
        ref_path = Path(repo_root) / ".git" / ref
        try:
            return ref_path.read_text().strip()
        except OSError:
            packed = Path(repo_root) / ".git" / "packed-refs"
            try:
                for line in packed.read_text().splitlines():
                    if line.endswith(f" {ref}"):
                        return line.split(" ", 1)[0]
            except OSError:
                return "unknown"
            return "unknown"
    return content


def directory_size_bytes(path: str | Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total
