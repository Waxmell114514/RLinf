"""Aggregate GRU mismatch runs across seeds and apply the pre-registered rule.

The thresholds below are copied from ``efference_probe/PREREG_wave3.md``
(section E1) and must not be edited after results exist; change the prereg
file, append a dated amendment, and re-derive instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# -- pre-registered (PREREG_wave3.md, E1) ----------------------------------
# Reproduction gate: the plan gates on exp4's ~56.5%.  exp3's ~12.3% is
# reported alongside but is informational only (the port reproduces exp4 to
# the decimal; see PREREG_wave3.md E1 for the exp3 discrepancy).
REFERENCE_RECALL = {"trace_abs": 56.5, "none": 12.3}
GATED_INTERFACES = ("trace_abs",)
RECALL_TOLERANCE = 5.0
REFERENCE_SEED = 42
# Smallest attribution effect we treat as real: the VLA paper's detectability
# limit (~0.04 balanced accuracy above the selection-aware floor), so the GRU
# and the 7B policy are judged by one yardstick.
MIN_EFFECT = 0.04
# The arm is informative only if the mechanics expose the mismatch.
POSITIVE_CONTROL_MIN = 0.10
SEED_QUORUM = 4
PRIMARY = ("trace_abs", "swap")


def _load(out_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    arms, probes = [], []
    for run_json in sorted(out_root.glob("*/seed*/run.json")):
        report = json.loads(run_json.read_text(encoding="utf-8"))
        protocol = report["protocol"]
        base = {
            "interface": protocol["interface"],
            "seed": int(protocol["seed"]),
            "scale": int(protocol["scale"]),
            "trailing_recall_pct": float(report["trailing_recall_pct"]),
        }
        if not report.get("arms"):
            arms.append({**base, "transform": ""})
        for transform, summary in report.get("arms", {}).items():
            row = {**base, "transform": transform}
            for key in (
                "n_samples",
                "n_positive",
                "n_groups",
                "selection_floor",
                "best_hidden",
                "best_hidden_any",
                "margin",
            ):
                row[key] = summary.get(key, np.nan)
            row["perm_p95"] = (summary.get("permutation_null_h") or {}).get(
                "p95", np.nan
            )
            arms.append(row)
        csv = run_json.parent / "probe_results.csv"
        if csv.is_file():
            table = pd.read_csv(csv)
            table["interface"] = base["interface"]
            table["seed"] = base["seed"]
            probes.append(table)
    return (
        pd.DataFrame(arms),
        pd.concat(probes, ignore_index=True) if probes else pd.DataFrame(),
    )


def _verdict(group: pd.DataFrame, probe_means: dict[str, float]) -> dict[str, Any]:
    n = len(group)
    readable = (group["margin"] >= MIN_EFFECT) & (
        group["best_hidden"] > group["perm_p95"].fillna(-np.inf)
    )
    control = probe_means.get("P2r", np.nan) - group["selection_floor"].mean()
    quorum = min(SEED_QUORUM, n)
    if not np.isfinite(control) or control < POSITIVE_CONTROL_MIN:
        verdict = "uninformative (positive control failed)"
    elif int(readable.sum()) >= quorum:
        verdict = "readable"
    elif int((~readable).sum()) >= quorum:
        verdict = "not readable"
    else:
        verdict = "mixed"
    return {
        "n_seeds": n,
        "seeds_readable": int(readable.sum()),
        "control_margin": float(control),
        "verdict": verdict,
    }


def _fmt(values: pd.Series, digits: int = 3) -> str:
    values = values.dropna()
    if values.empty:
        return "-"
    if len(values) == 1:
        return f"{values.iloc[0]:.{digits}f}"
    return f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}"


def write_summary(out_root: str | Path) -> Path:
    out_root = Path(out_root)
    arms, probes = _load(out_root)
    lines = ["# E1 — GRU mismatch: aggregate", ""]
    if arms.empty:
        lines.append("no runs found")
        path = out_root / "summary.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    # Reproduction gate.
    lines += ["## Reproduction gate (trailing recall, %)", ""]
    lines.append("| interface | seed 42 | all seeds | reference | gate |")
    lines.append("|---|---|---|---|---|")
    recall = arms.drop_duplicates(["interface", "seed"])
    for interface, group in recall.groupby("interface"):
        at_ref = group[group["seed"] == REFERENCE_SEED]["trailing_recall_pct"]
        reference = REFERENCE_RECALL.get(interface)
        gate = "-"
        if reference is not None and not at_ref.empty:
            scale = int(group["scale"].iloc[0])
            ok = abs(float(at_ref.iloc[0]) - reference) <= RECALL_TOLERANCE
            gate = ("PASS" if ok else "FAIL") + (" (quick mode)" if scale > 1 else "")
            if interface not in GATED_INTERFACES:
                gate = (
                    f"info only ({'within' if ok else 'outside'} ±{RECALL_TOLERANCE})"
                )
        at_ref_text = f"{at_ref.iloc[0]:.1f}" if not at_ref.empty else "-"
        lines.append(
            f"| {interface} | {at_ref_text} | "
            f"{_fmt(group['trailing_recall_pct'], digits=1)} | "
            f"{reference if reference is not None else '-'} | {gate} |"
        )
    lines += [
        "",
        f"Gate ({', '.join(GATED_INTERFACES)} only): seed {REFERENCE_SEED} "
        f"within ±{RECALL_TOLERANCE} points of the source's number. Quick-mode "
        "runs are smoke tests and cannot pass it.",
        "",
    ]

    arms = arms[arms["transform"] != ""]
    if arms.empty or probes.empty:
        path = out_root / "summary.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    rows = []
    for (interface, transform), group in arms.groupby(["interface", "transform"]):
        subset = probes[
            (probes["interface"] == interface) & (probes["transform"] == transform)
        ]
        means = subset.groupby("name")["balanced_acc_mean"].mean().to_dict()
        verdict = _verdict(group, means)
        row = {
            "interface": interface,
            "transform": transform,
            "n_samples": int(group["n_samples"].mean()),
            "floor": _fmt(group["selection_floor"]),
            "perm_p95": _fmt(group["perm_p95"]),
            "best_hidden (P1/P1g)": _fmt(group["best_hidden"]),
            "margin": _fmt(group["margin"]),
        }
        for name in ("P3", "P4", "P2", "P2r", "C_cmd", "C_out", "C_eff", "C_phase"):
            row[name] = _fmt(
                subset[subset["name"] == name].set_index("seed")["balanced_acc_mean"]
            )
        row.update(verdict)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out_root / "summary.csv", index=False)

    lines += ["## Attribution ladder per interface × transform", ""]
    lines.append(
        "Balanced accuracy, mean ± sd over seeds. `margin` = best hidden "
        "(max of P1, P1g) − selection floor (max of P0, P0g)."
    )
    lines.append("")
    columns = list(table.columns)
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "---|" * len(columns))
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")

    primary = table[
        (table["interface"] == PRIMARY[0]) & (table["transform"] == PRIMARY[1])
    ]
    lines += ["", "## Pre-registered readout", ""]
    lines.append(
        f"Rule: an arm is *readable* when, in ≥ {SEED_QUORUM} seeds, the best "
        f"hidden probe beats the selection floor by ≥ {MIN_EFFECT} and its "
        "permutation p95; it is *uninformative* when P2r clears the floor by "
        f"< {POSITIVE_CONTROL_MIN}."
    )
    if primary.empty:
        lines.append("")
        lines.append(f"Primary cell {PRIMARY} not run.")
    else:
        verdict = primary["verdict"].iloc[0]
        lines.append("")
        lines.append(f"Primary cell `{PRIMARY[0]}` × `{PRIMARY[1]}`: **{verdict}**.")
        mapping = {
            "readable": "keeping the command is enough in the GRU -> E5 (command "
            "input on the VLA) becomes the main line-A experiment",
            "not readable": "keeping the command is not enough -> a comparator is "
            "needed; E4 (world model as comparator) becomes the main experiment",
        }
        if verdict in mapping:
            lines.append(f"Plan consequence: {mapping[verdict]}.")
    path = out_root / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


__all__ = ["write_summary"]
