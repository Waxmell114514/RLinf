"""X4 readout (PREREG_X1_X4.md, X4 "Pre-registered readouts").

    P=experiments/efference_probe/x4_report.py
    python $P --data-root D/x4 --out experiments/efference_probe/reports/x4_report.md

Reads every completed ``x4_<ckpt>_<key>`` S2 run of ``x4_plan.yaml``.  It
reports, per checkpoint:

- B_mon = success(probe, monitor) - success(probe, off) and B_orc (oracle), with
  episode-bootstrap CIs resampled within condition;
- rules M and O against the pre-fixed yardstick G;
- the false-positive cost on clean episodes;
- online FPR/TPR in the off condition;
- undo accuracy;
- the sanity flag on the off arm.

Also holds the two smoke checks ``e4_wave.py``'s S0 gate uses for X4 runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from efference_probe import e4_common as common  # noqa: E402
from efference_probe.monitor import (  # noqa: E402
    CONDITIONS,
    LinearMonitor,
    mismatch_features,
)

PLAN_PATH = common.PROBE_DIR / "x4_plan.yaml"
# Fixed before collection from E4's pooled mirror s1 + s2 manifests.
G = {"sft": 0.167, "grpo": 0.219}
E4_PROBE_SUCCESS = {"sft": 0.477, "grpo": 0.730}
SANITY_TOLERANCE = 0.10
B, SEED = 2000, 0

__all__ = ["CONDITIONS", "readout", "recompute_gap", "undo_errors_cm"]


# -- smoke checks (used by e4_wave.gate_s0) ------------------------------------


def _stack(calls: pd.DataFrame, column: str) -> np.ndarray:
    return np.stack(calls[column].map(np.asarray).to_list())


def recompute_gap(
    calls: pd.DataFrame, monitor_cfg: dict[str, Any], schema: dict[str, Any]
) -> float:
    """Max |logged score - score recomputed offline from the logged rows|."""
    artifact = Path(monitor_cfg["artifact"])
    if not artifact.is_absolute():
        artifact = common.REPO_ROOT / artifact
    monitor = LinearMonitor.load(artifact, expect_sha256=monitor_cfg["artifact_sha256"])
    scored = calls[calls["monitor_score"].notna()]
    if scored.empty:
        return float("inf")
    chunks, dim = int(schema["num_action_chunks"]), int(schema["action_dim"])
    features = mismatch_features(
        _stack(scored, "a_cmd_env").reshape(-1, chunks, dim),
        _stack(scored, "states_before"),
        _stack(scored, "states_after"),
    )
    return float(np.max(np.abs(monitor.score(features) - scored["monitor_score"])))


def undo_errors_cm(calls: pd.DataFrame) -> np.ndarray:
    """Per undo of a hijacked chunk: |eef pos after the undo - eef pos before
    the hijacked call|, in cm (PREREG X4, "undo accuracy").

    Undos of self-commanded chunks (monitor false positives) have no hijacked
    call to return to and are left out.
    """
    errors = []
    for _, episode in calls.groupby("episode_id"):
        episode = episode.sort_values("call_idx").reset_index(drop=True)
        for i in np.flatnonzero(episode["is_recovery"].astype(bool).to_numpy()):
            if i == 0 or episode.loc[i - 1, "label"] != "HIJACK":
                continue
            after = np.asarray(episode.loc[i, "states_after"])[:3]
            target = np.asarray(episode.loc[i - 1, "states_before"])[:3]
            errors.append(100.0 * float(np.linalg.norm(after - target)))
    return np.asarray(errors)


# -- the readout -------------------------------------------------------------


def _bootstrap(
    treated: np.ndarray, control: np.ndarray, rng: np.random.Generator
) -> dict[str, float]:
    """Difference of success rates, episodes resampled within each group."""
    point = float(treated.mean() - control.mean())
    draws = [
        rng.choice(treated, len(treated)).mean()
        - rng.choice(control, len(control)).mean()
        for _ in range(B)
    ]
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "point": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_treated": int(len(treated)),
        "n_control": int(len(control)),
    }


def readout(
    ckpt: str, episodes: pd.DataFrame, calls: pd.DataFrame, threshold: float
) -> dict[str, Any]:
    """PREREG_X1_X4.md X4 readout for one checkpoint."""
    rng = np.random.default_rng(SEED)
    n_censored = int(episodes["censored"].astype(bool).sum())
    episodes = episodes[~episodes["censored"].astype(bool)]
    probe = episodes[episodes["is_probe_episode"].astype(bool)]
    clean = episodes[~episodes["is_probe_episode"].astype(bool)]

    def success(frame: pd.DataFrame, *conditions: str) -> np.ndarray:
        rows = frame[frame["monitor_condition"].isin(conditions)]
        return rows["success"].astype(float).to_numpy()

    g = G[ckpt]
    b_mon = _bootstrap(success(probe, "monitor"), success(probe, "off"), rng)
    b_orc = _bootstrap(success(probe, "oracle"), success(probe, "off"), rng)
    rule_m = b_mon["point"] >= 0.5 * g and b_mon["ci_low"] > 0
    rule_o = b_orc["point"] >= 0.5 * g and b_orc["ci_low"] > 0
    if not rule_o:
        reading = (
            "undo does not recover the cost; X4 is uninformative about the comparator"
        )
    elif rule_m:
        reading = "the external comparator closes at least half the hijack cost"
    else:
        reading = "recovery works; detection limits"
    off = calls[(calls["monitor_condition"] == "off") & calls["monitor_score"].notna()]
    self_scores = off.loc[off["label"] == "SELF", "monitor_score"]
    hijack_scores = off.loc[off["label"] == "HIJACK", "monitor_score"]
    off_success = (
        float(success(probe, "off").mean()) if len(success(probe, "off")) else None
    )
    errors = undo_errors_cm(calls)
    # Descriptive: a monitor false positive spends a call that could have been
    # hijacked, so monitor-arm probe episodes can carry slightly fewer hijacks.
    hijacks = {
        c: float(probe.loc[probe["monitor_condition"] == c, "n_hijacks"].mean())
        for c in CONDITIONS
        if "n_hijacks" in probe
    }
    return {
        "ckpt": ckpt,
        "G": g,
        "B_mon": b_mon,
        "B_orc": b_orc,
        "rule_M": bool(rule_m),
        "rule_O": bool(rule_o),
        "reading": reading,
        "fp_cost": _bootstrap(
            success(clean, "monitor"), success(clean, "off", "oracle"), rng
        ),
        "online_fpr": float((self_scores > threshold).mean())
        if len(self_scores)
        else None,
        "online_tpr": float((hijack_scores > threshold).mean())
        if len(hijack_scores)
        else None,
        "n_off_self_calls": int(len(self_scores)),
        "n_off_hijack_calls": int(len(hijack_scores)),
        "undo_error_median_cm": float(np.median(errors)) if len(errors) else None,
        "n_undos": int(len(errors)),
        "off_probe_success": off_success,
        "e4_probe_success": E4_PROBE_SUCCESS[ckpt],
        "sanity_flag": off_success is None
        or abs(off_success - E4_PROBE_SUCCESS[ckpt]) > SANITY_TOLERANCE,
        "hijacks_per_probe_episode": hijacks,
        "n_episodes": int(len(episodes)),
        "n_censored": n_censored,
    }


def load_ckpt(data_root: Path, ckpt: str, plan: dict[str, Any]):
    """Episodes and calls of every completed S2 run of one checkpoint."""
    episodes, calls, thresholds = [], [], set()
    for run in common.planned_runs(plan, "s2"):
        if run.ckpt != ckpt:
            continue
        run_id = common.completed_attempt(data_root, run.run_id)
        if not run_id:
            continue
        manifest = common.read_json(data_root / run_id / "manifest.json") or {}
        frame = pd.DataFrame(manifest.get("episodes", []))
        frame["run_id"] = run_id
        episodes.append(frame)
        table = pd.read_parquet(data_root / run_id / "calls.parquet")
        table["run_id"] = run_id
        # Episode ids restart per run; key them by run.
        table["episode_id"] = table["run_id"] + ":" + table["episode_id"].astype(str)
        calls.append(table)
        thresholds.add(manifest["monitor"]["threshold"])
    if not episodes:
        return None
    if len(thresholds) != 1:
        raise ValueError(f"{ckpt}: runs disagree on the monitor threshold {thresholds}")
    return pd.concat(episodes, ignore_index=True), pd.concat(calls), thresholds.pop()


def render(results: list[dict[str, Any]]) -> str:
    lines = [
        "# X4 report — external comparator in the loop",
        "",
        "Pre-registered in `PREREG_X1_X4.md` (X4). Every number is read from the runs' "
        "manifests and `calls.parquet`.",
        "",
        "| ckpt | G | B_mon [CI] | B_orc [CI] | rule M | rule O | reading |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        m, o = r["B_mon"], r["B_orc"]
        lines.append(
            f"| {r['ckpt']} | {r['G']:.3f} | {m['point']:+.3f} [{m['ci_low']:+.3f}, "
            f"{m['ci_high']:+.3f}] | {o['point']:+.3f} [{o['ci_low']:+.3f}, "
            f"{o['ci_high']:+.3f}] | {r['rule_M']} | {r['rule_O']} | {r['reading']} |"
        )
    lines += ["", "```json", json.dumps(results, indent=2, default=str), "```", ""]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    plan = common.load_plan(args.plan)
    results = []
    for ckpt in common.CKPTS:
        loaded = load_ckpt(args.data_root, ckpt, plan)
        if loaded is not None:
            results.append(readout(ckpt, *loaded))
    text = render(results)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
