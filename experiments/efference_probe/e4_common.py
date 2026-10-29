"""Shared pieces for the E4 wave: the run plan, result readers, bookkeeping.

Pure numpy/pandas, no torch: the driver (``e4_wave.py``) and the report
(``e4_report.py``) both run on CPU-only boxes.

Bookkeeping is one small file per event (a status marker per finished job, a
ledger record per subprocess) rather than shared append-only files, because
the wave may be driven from several containers writing one network volume,
and concurrent appends to one file there are last-writer-wins.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

PROBE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROBE_DIR.parents[1]
CONFIG_DIR = PROBE_DIR / "configs"
PLAN_PATH = PROBE_DIR / "e4_plan.yaml"
CKPTS = ("sft", "grpo")

# The pre-registered readout must be in the canonical logbook before S2 runs,
# and so must every amendment the S2 code paths depend on.
PREREG_MARKER = "E4 PRE-REGISTERED READOUT"
AMENDMENT_MARKERS = ("E4 PRE-REGISTRATION AMENDMENT 1",)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_plan(path: Path = PLAN_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# -- the run plan ------------------------------------------------------------


@dataclass(frozen=True)
class PlannedRun:
    """One collection: a plan entry bound to a checkpoint."""

    stage: str
    ckpt: str
    key: str
    config: str
    expected_s: float
    suite: str = "goal"
    transform: str = ""
    seed: Optional[int] = None
    decoding: bool = True
    overrides: tuple[str, ...] = field(default_factory=tuple)
    # The plan's ``wave``: E4, or a later wave (X1, X4) on the same driver.
    wave: str = "e4"

    @property
    def run_id(self) -> str:
        """The planned id; a retry after an interruption gets ``_r2``, ``_r3``."""
        return f"{self.wave}_{self.ckpt}_{self.key}"


def planned_runs(plan: dict[str, Any], stage: str) -> list[PlannedRun]:
    """Expand a stage into runs, interleaving checkpoints entry by entry.

    Interleaving keeps the SFT/GRPO contrast balanced if a budget stop cuts the
    wave short: both checkpoints have the primary arms before either starts
    the next one.
    """
    runs = []
    # A plan may put its checkpoints in another order (X4 runs GRPO first).
    order = plan.get("ckpt_order", CKPTS)
    if sorted(order) != sorted(CKPTS):
        raise ValueError(f"ckpt_order must be a permutation of {CKPTS}: {order}")
    for entry in plan["stages"][stage]:
        for ckpt in order:
            runs.append(
                PlannedRun(
                    stage=stage,
                    ckpt=ckpt,
                    key=entry["key"],
                    config=entry["config"],
                    expected_s=float(entry["expected_s"]),
                    suite=entry.get("suite", "goal"),
                    transform=entry.get("transform", ""),
                    seed=entry.get("seed"),
                    decoding=bool(entry.get("decoding", True)),
                    overrides=tuple(entry.get("overrides", ())),
                    wave=plan.get("wave", "e4"),
                )
            )
    return runs


def find_planned(plan: dict[str, Any], ckpt: str, key: str) -> PlannedRun:
    for stage in plan["stages"]:
        for run in planned_runs(plan, stage):
            if run.ckpt == ckpt and run.key == key:
                return run
    raise KeyError(f"no planned run {ckpt}/{key} in the E4 plan")


def attempt_ids(data_root: Path, planned_id: str) -> list[str]:
    """Run directories that exist for a planned id, in attempt order."""
    found = []
    for path in data_root.glob(f"{planned_id}*"):
        match = re.fullmatch(re.escape(planned_id) + r"(?:_r(\d+))?", path.name)
        if match and path.is_dir():
            found.append((int(match.group(1) or 1), path.name))
    return [name for _, name in sorted(found)]


def next_attempt_id(data_root: Path, planned_id: str) -> str:
    """First id whose directory does not exist yet -- never reuse a run dir."""
    attempts = attempt_ids(data_root, planned_id)
    if not attempts:
        return planned_id
    last = attempts[-1]
    match = re.fullmatch(re.escape(planned_id) + r"(?:_r(\d+))?", last)
    return f"{planned_id}_r{int(match.group(1) or 1) + 1}"


# -- status markers and the ledger --------------------------------------------


def status_dir(data_root: Path) -> Path:
    return data_root / ".e4_status"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def status_path(data_root: Path, run_id: str, kind: str) -> Path:
    return status_dir(data_root) / f"{run_id}.{kind}.json"


def job_status(data_root: Path, run_id: str, kind: str) -> Optional[dict[str, Any]]:
    """Completion marker for ``kind`` (collect, probe-linear, probe-mlp, ...).

    Written by the driver only after the child exited 0 *and* the content
    checks passed.  ``manifest.json`` is not a completion signal: the harness
    rewrites it after every batch, and the partial copy already reads
    ``stop_reason: completed``.
    """
    return read_json(status_path(data_root, run_id, kind))


def completed_attempt(
    data_root: Path, planned_id: str, kind: str = "collect"
) -> Optional[str]:
    for run_id in attempt_ids(data_root, planned_id):
        status = job_status(data_root, run_id, kind)
        if status and status.get("ok"):
            return run_id
    return None


def ledger_dir(data_root: Path) -> Path:
    return data_root / "e4_ledger"


def ledger_records(data_root: Path) -> pd.DataFrame:
    rows = [read_json(p) for p in sorted(ledger_dir(data_root).glob("*.json"))]
    frame = pd.DataFrame([r for r in rows if r])
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "start_utc",
                "end_utc",
                "kind",
                "run_id",
                "rc",
                "elapsed_s",
                "outcome",
            ]
        )
    return frame.sort_values("start_utc").reset_index(drop=True)


def gpu_hours_used(data_root: Path) -> float:
    """Collection wall clock summed over the wave, failed attempts included."""
    frame = ledger_records(data_root)
    if frame.empty:
        return 0.0
    return float(frame.loc[frame["kind"] == "collect", "elapsed_s"].sum()) / 3600.0


# -- analysis readers ----------------------------------------------------------


def analysis_dir(run_dir: Path, family: str) -> Path:
    return run_dir / (
        "analysis" if family in ("linear", "pilot") else f"analysis_{family}"
    )


def summary_extra(path: Path) -> dict[str, Any]:
    """The JSON "Extra" block that ``analysis.summarize`` appends to summary.md."""
    # A summary written on Windows has CRLF line ends.
    text = (path / "summary.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    _, _, tail = text.partition("#### Extra")
    match = re.search(r"```json\n(.*?)\n```", tail, flags=re.S)
    return json.loads(match.group(1)) if match else {}


def probe_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path / "probe_results.csv")


def best_cell(frame: pd.DataFrame, name: str) -> Optional[dict[str, Any]]:
    """A rung at its best (layer, pool) cell -- the convention the paper reports."""
    rows = frame[frame["name"] == name]
    if rows.empty:
        return None
    row = rows.loc[rows["balanced_acc_mean"].idxmax()]
    cell = (
        f"L{int(row['layer'])}/{row['pool']}"
        if pd.notna(row.get("layer")) and pd.notna(row.get("pool"))
        else "-"
    )
    return {
        "bacc": float(row["balanced_acc_mean"]),
        "sd": float(row["balanced_acc_std"]),
        "auroc": float(row["auroc_mean"]),
        "cell": cell,
        "n_cells": int(len(rows)),
        "mean_over_cells": float(rows["balanced_acc_mean"].mean()),
    }


def e1_best(path: Path) -> Optional[dict[str, Any]]:
    csv = path / "e1_readout.csv"
    if not csv.is_file():
        return None
    frame = pd.read_csv(csv)
    if frame.empty:
        return None
    row = frame.loc[frame["r2_mean"].idxmax()]
    return {"r2": float(row["r2_mean"]), "cell": f"L{int(row['layer'])}/{row['pool']}"}


def hidden_integrity(
    run_dir: Path, expect_tail: Optional[tuple[int, ...]] = None
) -> dict:
    """Open every hidden archive: shape, dtype, finiteness, total call count.

    The total matching ``calls.parquet``'s row count is the strong check -- a
    truncated or damaged archive breaks it (wave 2 lost a run to 0-byte npz).
    """
    total, bad, tails = 0, [], set()
    files = sorted((run_dir / "hidden").glob("ep*.npz"))
    for path in files:
        try:
            with np.load(path) as archive:
                h = archive["h"]
                if h.ndim != 4 or not np.isfinite(h).all() or h.dtype != np.float16:
                    bad.append(path.name)
                tails.add(tuple(int(x) for x in h.shape[1:]))
                total += int(h.shape[0])
        except Exception as error:  # noqa: BLE001 - any unreadable archive is "bad"
            bad.append(f"{path.name}: {type(error).__name__}")
    report = {
        "n_archives": len(files),
        "n_hidden_calls": total,
        "bad_archives": bad,
        "shape_tails": sorted(tails),
    }
    if expect_tail is not None:
        report["shape_ok"] = tails == {tuple(expect_tail)}
    return report


# -- per-task competence (paper T4) --------------------------------------------

# Pre-registered in the logbook with the E4 readout.  A flag is a reporting
# aid: flagged tasks stay in every analysis.
LOW_SUCCESS = 0.10
RELATIVE_WEAK = 0.5


def competence_report(run_dir: Path) -> pd.DataFrame:
    """Per-task success and intervention strength, from logged data only.

    On a task the policy cannot do it barely moves and emits near-identical
    chunks, so SWAP executes almost what was commanded (paper T4).  Two
    strength measures, both on the 6 motion dims of all 8 substeps:

    * ``intervention_rel_change`` -- median over hijacked calls of
      ||a_exec - a_cmd|| / ||a_cmd||: how different the executed chunk was.
    * ``self_motion`` -- median end-effector displacement on SELF calls.
    """
    calls = pd.read_parquet(run_dir / "calls.parquet")
    manifest = read_json(run_dir / "manifest.json") or {}
    episodes = pd.DataFrame(manifest.get("episodes", []))
    if not episodes.empty and "censored" in episodes:
        episodes = episodes[~episodes["censored"].astype(bool)]

    def motion_dims(column: str) -> np.ndarray:
        chunks = np.stack([np.asarray(v, dtype=np.float32) for v in calls[column]])
        return chunks.reshape(len(calls), -1, 7)[:, :, :6].reshape(len(calls), -1)

    commanded, executed = motion_dims("a_cmd_env"), motion_dims("a_exec")
    before = np.stack([np.asarray(v, dtype=np.float32) for v in calls["states_before"]])
    after = np.stack([np.asarray(v, dtype=np.float32) for v in calls["states_after"]])
    calls = calls.assign(
        rel_change=np.linalg.norm(executed - commanded, axis=1)
        / np.maximum(np.linalg.norm(commanded, axis=1), 1e-8),
        motion=np.linalg.norm(after[:, :3] - before[:, :3], axis=1),
    )

    rows = []
    for task_id, task_calls in calls.groupby("task_id"):
        hijacked = task_calls[task_calls["label"] == "HIJACK"]
        selfs = task_calls[task_calls["label"] == "SELF"]
        task_eps = (
            episodes[episodes["task_id"] == task_id] if not episodes.empty else episodes
        )

        def rate(subset: pd.DataFrame) -> float:
            return float(subset["success"].mean()) if len(subset) else float("nan")

        rows.append(
            {
                "task_id": int(task_id),
                "n_episodes": int(len(task_eps)),
                "success_rate": rate(task_eps),
                "success_probe": rate(
                    task_eps[task_eps["is_probe_episode"].astype(bool)]
                )
                if len(task_eps)
                else float("nan"),
                "success_clean": rate(
                    task_eps[~task_eps["is_probe_episode"].astype(bool)]
                )
                if len(task_eps)
                else float("nan"),
                "n_hijacks": int(len(hijacked)),
                "intervention_rel_change": float(hijacked["rel_change"].median())
                if len(hijacked)
                else float("nan"),
                "self_motion": float(selfs["motion"].median())
                if len(selfs)
                else float("nan"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["flag_low_success"] = frame["success_rate"] <= LOW_SUCCESS
    # Relative flags need a meaningful across-task median.
    relative = len(frame) >= 3
    for column, flag in (
        ("intervention_rel_change", "flag_weak_intervention"),
        ("self_motion", "flag_low_motion"),
    ):
        median = frame[column].median()
        frame[flag] = (frame[column] <= RELATIVE_WEAK * median) if relative else False
    frame["flagged"] = frame[
        ["flag_low_success", "flag_weak_intervention", "flag_low_motion"]
    ].any(axis=1)
    return frame


def derived_dir(data_root: Path, run_id: str) -> Path:
    """E4-derived outputs live outside the run directory, which stays as collected."""
    return data_root / "e4_derived" / run_id
