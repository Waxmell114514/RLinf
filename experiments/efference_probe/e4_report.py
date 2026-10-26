#!/usr/bin/env python
"""Build ``reports/e4_report.md`` from the E4 run directories.

    python experiments/efference_probe/e4_report.py \
        --data-root experiments/efference_probe/data \
        --existing-root /path/to/wave2/data        # optional

Every number is read from a results file.  Existing wave-2 rows come from
their own ``analysis*/`` directories when ``--existing-root`` has them;
otherwise from the committed ``reports/p0_wave2_report.md`` (marked †, and E1
is then ``[TO VERIFY]`` because that report does not carry it).  Anything not
found is printed as ``[TO VERIFY: ...]`` -- never estimated.

The verdicts apply the rules pre-registered in ``logbook.md`` under
``E4 PRE-REGISTERED READOUT`` (R1-R12); the thresholds below are those rules,
not tuning knobs.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from efference_probe import e4_common as common  # noqa: E402

# Pre-registered constants (logbook, E4 PRE-REGISTERED READOUT).
DELTA = 0.04  # R1/R2: margin over floor / over C_dstates
INFORMATIVE_P2R = 0.10  # R3: P2r must clear the floor by this much
CMD_BAND = (0.45, 0.55)  # R2: C_cmd at chance
E1_MIN_R2 = 0.30  # R9
LOW_N = 600  # R11
ALPHA = 0.05  # R7
RUNGS = ("P1", "P3", "P4")
SUITES = ("goal", "spatial", "object", "long")
FAMILIES = ("linear", "mlp")
WAVE2_REPORT = common.PROBE_DIR / "reports" / "p0_wave2_report.md"
TO_VERIFY = "[TO VERIFY]"


@dataclass
class RunNumbers:
    """Everything one table row needs, plus where it came from."""

    label: str
    run_id: str
    suite: str
    transform: str
    seed: Optional[int]
    source: str  # "analysis", "wave2 report", or "missing: <why>"
    families: dict[str, dict[str, Any]] = field(default_factory=dict)
    e1: Optional[dict[str, Any]] = None
    e3: dict[str, Any] = field(default_factory=dict)
    success: dict[str, Any] = field(default_factory=dict)
    n: Optional[int] = None
    flagged_tasks: Optional[list[int]] = None

    @property
    def dagger(self) -> str:
        return "†" if self.source == "wave2 report" else ""


# -- reading -------------------------------------------------------------------


def _family_numbers(path: Path) -> dict[str, Any]:
    frame = common.probe_frame(path)
    extra = common.summary_extra(path)
    numbers = {
        name: common.best_cell(frame, name)
        for name in ("P0", *RUNGS, "P2", "P2r", "C_cmd", "C_dstates", "C_phase")
    }
    numbers["floor"] = extra.get("selection_floor", {}).get("max_over_cells")
    numbers["perm_p95"] = extra.get("permutation_null", {}).get("p95")
    numbers["_extra"] = extra
    return numbers


def read_run(
    run_dir: Path, label: str, suite: str, transform: str, seed, data_root=None
) -> RunNumbers:
    """Numbers for one run straight from its analysis directories."""
    row = RunNumbers(label, run_dir.name, suite, transform, seed, source="analysis")
    for family in FAMILIES:
        path = common.analysis_dir(run_dir, family)
        if (path / "probe_results.csv").is_file() and (path / "summary.md").is_file():
            row.families[family] = _family_numbers(path)
    if not row.families:
        row.source = "missing: no completed analysis directory"
        return row
    extra = next(iter(row.families.values()))["_extra"]
    row.e1 = common.e1_best(common.analysis_dir(run_dir, "linear"))
    e3 = extra.get("e3", {})
    row.e3 = {
        "d_entropy": e3.get("entropy_mean", {}).get("paired_mean_difference"),
        "p_entropy": e3.get("entropy_mean", {}).get("wilcoxon_p"),
        "d_logprob": e3.get("logprob_sum", {}).get("paired_mean_difference"),
        "p_logprob": e3.get("logprob_sum", {}).get("wilcoxon_p"),
        "n_ep": e3.get("entropy_mean", {}).get("paired_n_episodes"),
    }
    rates = e3.get("success_rate", {})
    row.success = {
        "overall": rates.get("overall"),
        "probe": rates.get("probe_episodes", {}).get("success_rate"),
        "clean": rates.get("clean_episodes", {}).get("success_rate"),
    }
    if e3:
        row.n = int(e3.get("n_positive", 0)) + int(e3.get("n_negative", 0))
    if data_root is not None:
        competence = common.derived_dir(data_root, run_dir.name) / "competence.csv"
        if competence.is_file():
            import pandas as pd

            frame = pd.read_csv(competence)
            row.flagged_tasks = frame.loc[
                frame["flagged"].astype(bool), "task_id"
            ].tolist()
    return row


def _num(text: str) -> float:
    return float(text.replace("+", ""))


def read_wave2_report(
    run_id: str, label: str, suite: str, transform: str, seed, path: Path = WAVE2_REPORT
) -> RunNumbers:
    """Fallback: the numbers the wave-2 agent transcribed from that run's files."""
    row = RunNumbers(label, run_id, suite, transform, seed, source="wave2 report")
    text = path.read_text(encoding="utf-8")
    for family in FAMILIES:
        head = re.search(
            rf"\*\*{re.escape(run_id)} / {family}\*\* — selection floor "
            r"max_over_cells=([\d.]+)([^\n]*)\n\n(.*?)\n\n",
            text,
            flags=re.S,
        )
        if not head:
            continue
        numbers: dict[str, Any] = {"floor": float(head.group(1)), "perm_p95": None}
        p95 = re.search(r"p95=([\d.]+)", head.group(2))
        if p95:
            numbers["perm_p95"] = float(p95.group(1))
        for line in head.group(3).splitlines():
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) == 6 and re.fullmatch(r"P\d|P2r|C_\w+", cells[0]):
                numbers[cells[0]] = {
                    "bacc": float(cells[1]),
                    "sd": float(cells[2]),
                    "auroc": float(cells[3]),
                    "cell": cells[4],
                    "n_cells": int(cells[5]),
                }
        row.families[family] = numbers
    e3 = re.search(
        rf"^\| {re.escape(run_id)} \| ([+-][\d.]+) \| ([\d.e-]+) \| ([+-][\d.]+) \| ([\d.e-]+) \| (\d+)",
        text,
        flags=re.M,
    )
    if e3:
        row.e3 = {
            "d_entropy": _num(e3.group(1)),
            "p_entropy": float(e3.group(2)),
            "d_logprob": _num(e3.group(3)),
            "p_logprob": float(e3.group(4)),
            "n_ep": int(e3.group(5)),
        }
    success = re.search(
        rf"^\| {re.escape(run_id)} \| ([\d.]+) \| ([\d.]+) \(\d+\) \| ([\d.]+) \(\d+\) \|",
        text,
        flags=re.M,
    )
    if success:
        row.success = {
            "overall": float(success.group(1)),
            "probe": float(success.group(2)),
            "clean": float(success.group(3)),
        }
    samples = re.search(
        rf"^\| {re.escape(run_id)} (?:mlp|linear) \| (\d+) \|", text, flags=re.M
    )
    if samples:
        row.n = int(samples.group(1))
    if not row.families:
        row.source = "missing: not in the wave-2 report"
    return row


# -- the pre-registered rules ------------------------------------------------


def ladder_verdict(row: RunNumbers, family: str) -> dict[str, Any]:
    """R1-R3 for one run and family."""
    numbers = row.families.get(family)
    if not numbers or numbers.get("floor") is None:
        return {"status": "missing"}
    floor = numbers["floor"]
    p2r = (numbers.get("P2r") or {}).get("bacc")
    dstates = (numbers.get("C_dstates") or {}).get("bacc")
    cmd = (numbers.get("C_cmd") or {}).get("bacc")
    margins = {r: numbers[r]["bacc"] - floor for r in RUNGS if numbers.get(r)}
    if len(margins) < len(RUNGS) or p2r is None or dstates is None or cmd is None:
        return {"status": "missing", "margins": margins}
    if p2r < floor + INFORMATIVE_P2R:
        return {
            "status": "uninformative",
            "margins": margins,
            "p2r": p2r,
            "floor": floor,
        }
    above = [r for r, m in margins.items() if m >= DELTA]
    cmd_ok = CMD_BAND[0] <= cmd <= CMD_BAND[1]
    clears_dstates = [r for r in above if numbers[r]["bacc"] >= dstates + DELTA]
    beyond = clears_dstates if cmd_ok else []
    # Which R2 condition an above-floor run failed; either one alone suffices.
    explained_by = []
    if above and not clears_dstates:
        explained_by.append("C_dstates")
    if above and not cmd_ok:
        explained_by.append("C_cmd")
    return {
        "status": "above" if above else "at",
        "above": above,
        "beyond_observables": beyond,
        "explained_by": explained_by,
        "margins": margins,
        "floor": floor,
        "p2r": p2r,
        "c_dstates": dstates,
        "c_cmd": cmd,
    }


_EXPLAINED_NAMES = {"C_dstates": "C_dstates", "C_cmd": "C_cmd off chance"}


def explained_label(v: dict[str, Any]) -> str:
    """The R2 condition(s) that kept an above-floor run from counting beyond observables."""
    parts = []
    if "C_dstates" in v["explained_by"]:
        parts.append(f"C_dstates {v['c_dstates']:.3f}")
    if "C_cmd" in v["explained_by"]:
        parts.append(f"C_cmd {v['c_cmd']:.3f} outside [{CMD_BAND[0]}, {CMD_BAND[1]}]")
        if "C_dstates" not in v["explained_by"]:
            parts.append(f"rungs clear C_dstates {v['c_dstates']:.3f} + δ")
    return "none (" + "; ".join(parts) + ")"


def checkpoint_verdict(seed_rows: list[Optional[RunNumbers]], family: str) -> str:
    """R4: both goal-swap seeds must agree, or the answer is inconclusive."""
    if len(seed_rows) != 2 or any(r is None for r in seed_rows):
        return "inconclusive (a seed is missing)"
    verdicts = [ladder_verdict(r, family) for r in seed_rows]
    statuses = [v["status"] for v in verdicts]
    if "missing" in statuses:
        return "inconclusive (numbers missing)"
    if "uninformative" in statuses:
        return "inconclusive (a run is uninformative under R3)"
    if all(v["status"] == "above" and v["beyond_observables"] for v in verdicts):
        return "above floor"
    if all(v["status"] == "at" for v in verdicts):
        return "at floor"
    explained = [
        v for v in verdicts if v["status"] == "above" and not v["beyond_observables"]
    ]
    if explained:
        why = sorted({w for v in explained for w in v["explained_by"]})
        return (
            "inconclusive (above floor but explained by outcome perception in a"
            f" seed: {' and '.join(_EXPLAINED_NAMES[w] for w in why)})"
        )
    return "inconclusive (seeds disagree)"


def contrast(sft: str, grpo: str) -> str:
    """R5."""
    if "inconclusive" in sft or "inconclusive" in grpo:
        return "(e) inconclusive: a checkpoint has no replicated verdict"
    table = {
        (
            "at floor",
            "above floor",
        ): "(a) RL post-training creates an attribution-relevant "
        "representation that SFT lacks",
        (
            "at floor",
            "at floor",
        ): "(b) RL post-training does not create a readable attribution trace",
        (
            "above floor",
            "above floor",
        ): "(c) a readable trace is present in both; not RL-specific",
        ("above floor", "at floor"): "(d) Base-Lora above floor, GRPO at floor",
    }
    return table[(sft, grpo)]


def _surprised(row: Optional[RunNumbers]) -> Optional[bool]:
    if row is None or row.e3.get("p_entropy") is None:
        return None
    e3 = row.e3
    return bool(
        (e3["d_entropy"] > 0 and e3["p_entropy"] < ALPHA)
        or (e3["d_logprob"] < 0 and e3["p_logprob"] < ALPHA)
    )


def e3_migration(sft_freeze: list, grpo_freeze: list) -> str:
    """R7."""
    grpo = [_surprised(r) for r in grpo_freeze]
    sft = [_surprised(r) for r in sft_freeze]
    if None in grpo or None in sft or len(grpo) != 2 or len(sft) != 2:
        return f"{TO_VERIFY}: a freeze arm is missing"
    if all(grpo) and not any(sft):
        return "migrated toward mismatch-gated (GRPO reacts to freeze on both seeds; Base-Lora on neither)"
    return (
        f"not migrated (GRPO freeze reaction on {sum(grpo)}/2 seeds, "
        f"Base-Lora on {sum(sft)}/2)"
    )


def compensation(sft_rows: list, grpo_rows: list) -> str:
    """R8, over the swap and mirror seeds paired by key."""
    gaps = []
    for sft, grpo in zip(sft_rows, grpo_rows):
        values = [
            r.success.get("clean") - r.success.get("probe")
            if r
            and r.success.get("clean") is not None
            and r.success.get("probe") is not None
            else None
            for r in (sft, grpo)
        ]
        gaps.append(values)
    if any(None in g for g in gaps) or len(gaps) != 4:
        return f"{TO_VERIFY}: a swap/mirror arm or its probe/clean success split is missing"
    stronger = all(grpo < sft for sft, grpo in gaps)
    detail = ", ".join(f"{s:+.2f}->{g:+.2f}" for s, g in gaps)
    return (
        "strengthens" if stronger else "does not strengthen on every arm"
    ) + f" (clean-probe gap Base-Lora->GRPO: {detail})"


# -- rendering -------------------------------------------------------------------


def _bacc(cell: Optional[dict[str, Any]]) -> str:
    return f"{cell['bacc']:.3f}±{cell['sd']:.3f}" if cell else TO_VERIFY


def _p(value: Optional[float]) -> str:
    return (
        TO_VERIFY
        if value is None
        else (f"{value:.2g}" if value < 0.01 else f"{value:.3f}")
    )


def table_rows(rows: list[RunNumbers], family: str) -> list[str]:
    header = (
        "| row | run | seed | n | P0 floor | P1 | P3 | P4 | P2r | C_dstates | C_cmd | "
        "E1 R² (cell) | E3 Δentropy (p) | E3 Δlogprob (p) | success all / probe / clean | n_ep |"
    )
    lines = [header, "|" + "---|" * 16]
    for row in rows:
        numbers = row.families.get(family)
        if row.source.startswith("missing") or numbers is None:
            reason = (
                row.source
                if row.source.startswith("missing")
                else f"missing: no {family} round"
            )
            lines.append(
                f"| {row.label} | `{row.run_id}` | {row.seed} | "
                + " | ".join([f"[TO VERIFY: {reason}]"] + [""] * 12)
                + " |"
            )
            continue
        d = row.dagger
        if family == "linear":
            e1 = (
                f"{row.e1['r2']:.3f} ({row.e1['cell']})"
                if row.e1
                else (f"{TO_VERIFY}: not in the wave-2 report" if d else TO_VERIFY)
            )
        else:
            e1 = "n/a (linear-only)"
        e3 = row.e3
        ent = (
            f"{e3['d_entropy']:+.4f} ({_p(e3['p_entropy'])})"
            if e3.get("d_entropy") is not None
            else TO_VERIFY
        )
        lp = (
            f"{e3['d_logprob']:+.3f} ({_p(e3['p_logprob'])})"
            if e3.get("d_logprob") is not None
            else TO_VERIFY
        )
        succ = row.success
        success = (
            " / ".join(
                f"{succ[k]:.2f}" if succ.get(k) is not None else "?"
                for k in ("overall", "probe", "clean")
            )
            if succ
            else TO_VERIFY
        )
        floor = (
            f"{numbers['floor']:.3f}" if numbers.get("floor") is not None else TO_VERIFY
        )
        lines.append(
            f"| {row.label}{d} | `{row.run_id}` | {row.seed} | {row.n or TO_VERIFY} | {floor} | "
            + " | ".join(
                _bacc(numbers.get(k))
                for k in ("P1", "P3", "P4", "P2r", "C_dstates", "C_cmd")
            )
            + f" | {e1} | {ent} | {lp} | {success} | {e3.get('n_ep') or TO_VERIFY} |"
        )
    return lines


def collect_rows(
    plan: dict[str, Any], data_root: Path, existing_root: Optional[Path]
) -> list[RunNumbers]:
    rows = []
    for entry in plan["existing"]["runs"]:
        args = (entry["label"], entry["suite"], entry["transform"], entry["seed"])
        run_dir = existing_root / entry["run_id"] if existing_root else None
        if run_dir is not None and (run_dir / "analysis" / "summary.md").is_file():
            rows.append(read_run(run_dir, *args))
        else:
            rows.append(read_wave2_report(entry["run_id"], *args))
    for run in common.planned_runs(plan, "s2"):
        label = plan["checkpoints"][run.ckpt]["label"]
        run_id = common.completed_attempt(data_root, run.run_id)
        if run_id is None:
            rows.append(
                RunNumbers(
                    label,
                    run.run_id,
                    run.suite,
                    run.transform,
                    run.seed,
                    "missing: not collected",
                )
            )
        else:
            rows.append(
                read_run(
                    data_root / run_id,
                    label,
                    run.suite,
                    run.transform,
                    run.seed,
                    data_root,
                )
            )
    return rows


def _pick(rows, label, suite, transform, seed) -> Optional[RunNumbers]:
    for row in rows:
        if (row.label, row.suite, row.transform, row.seed) == (
            label,
            suite,
            transform,
            seed,
        ):
            return None if row.source.startswith("missing") else row
    return None


def control_arms(rows: list[RunNumbers], sft: str, grpo: str) -> str:
    """R6: the gross-violation arms count only through the C_dstates twin."""
    beyond = {}
    for label in (sft, grpo):
        for transform in ("mirror", "freeze"):
            seeds = [_pick(rows, label, "goal", transform, s) for s in (1, 2)]
            if any(r is None for r in seeds):
                return f"{TO_VERIFY}: a {label} {transform} seed is missing"
            verdicts = [ladder_verdict(r, "mlp") for r in seeds]
            if any(v["status"] == "missing" for v in verdicts):
                return f"{TO_VERIFY}: a {label} {transform} MLP round is missing"
            beyond[(label, transform)] = all(
                v["status"] == "above" and v["beyond_observables"] for v in verdicts
            )
    supports = [
        t for t in ("mirror", "freeze") if beyond[(grpo, t)] and not beyond[(sft, t)]
    ]
    detail = ", ".join(
        f"{label} {t}: {'beyond' if beyond[(label, t)] else 'not beyond'}"
        for label in (sft, grpo)
        for t in ("mirror", "freeze")
    )
    if supports:
        return f"supports (a) via {supports} ({detail})"
    return f"no evidence for attribution from these arms ({detail})"


def readout(plan: dict[str, Any], rows: list[RunNumbers]) -> list[str]:
    sft, grpo = (
        plan["checkpoints"]["sft"]["label"],
        plan["checkpoints"]["grpo"]["label"],
    )
    lines = ["## Pre-registered readout, applied", ""]
    verdicts = {}
    for family in ("mlp", "linear"):
        for label in (sft, grpo):
            seeds = [_pick(rows, label, "goal", "swap", s) for s in (1, 2)]
            verdicts[(family, label)] = checkpoint_verdict(seeds, family)
            # R9: a failed positive control withholds the checkpoint's verdict.
            if any(r and r.e1 and r.e1["r2"] < E1_MIN_R2 for r in seeds):
                verdicts[(family, label)] = (
                    "inconclusive (withheld under R9: E1 R² < 0.30)"
                )
        primary = "primary" if family == "mlp" else "secondary"
        lines += [
            f"**R4/R5, {family} family ({primary}).** {sft}: {verdicts[(family, sft)]}; "
            f"{grpo}: {verdicts[(family, grpo)]}. Contrast: "
            f"{contrast(verdicts[(family, sft)], verdicts[(family, grpo)])}.",
            "",
        ]
    lines.append(
        "**Per-run ladders (R1-R3, R11).** δ = 0.04 over the same-family floor; "
        "'beyond observables' also needs ≥ C_dstates + δ and C_cmd at chance (R2)."
    )
    lines.append("")
    for row in rows:
        for family in FAMILIES:
            if family not in row.families:
                continue
            v = ladder_verdict(row, family)
            if v["status"] == "missing":
                continue
            margins = ", ".join(f"{k} {m:+.3f}" for k, m in v["margins"].items())
            extra = ""
            if v["status"] == "above":
                extra = f"; beyond observables: {v['beyond_observables'] or explained_label(v)}"
            low_n = " — low n" if row.n is not None and row.n < LOW_N else ""
            post_hoc = (
                " (wave-2 row; rule applied for reference)"
                if row.label not in (sft, grpo)
                else (" (R10: single-seed suite)" if row.suite != "goal" else "")
            )
            lines.append(
                f"- `{row.run_id}`{row.dagger} {family}: **{v['status']}** (margins {margins}){extra}{low_n}{post_hoc}"
            )
    lines.append("")
    lines += [f"**R6 (mirror/freeze, MLP).** {control_arms(rows, sft, grpo)}.", ""]
    sft_freeze = [_pick(rows, sft, "goal", "freeze", s) for s in (1, 2)]
    grpo_freeze = [_pick(rows, grpo, "goal", "freeze", s) for s in (1, 2)]
    lines += [f"**R7 (E3 migration).** {e3_migration(sft_freeze, grpo_freeze)}.", ""]
    keys = [("swap", 1), ("swap", 2), ("mirror", 1), ("mirror", 2)]
    lines += [
        "**R8 (compensation).** "
        + compensation(
            [_pick(rows, sft, "goal", t, s) for t, s in keys],
            [_pick(rows, grpo, "goal", t, s) for t, s in keys],
        )
        + ".",
        "",
    ]
    e1_notes = []
    for label in (sft, grpo):
        for seed in (1, 2):
            row = _pick(rows, label, "goal", "swap", seed)
            if row is None or row.e1 is None:
                e1_notes.append(f"{label} s{seed}: {TO_VERIFY}")
            else:
                ok = (
                    ""
                    if row.e1["r2"] >= E1_MIN_R2
                    else " — BELOW 0.30, verdict withheld"
                )
                e1_notes.append(
                    f"{label} s{seed}: {row.e1['r2']:.3f} ({row.e1['cell']}){ok}"
                )
    lines += ["**R9 (E1 positive control).** " + "; ".join(e1_notes) + ".", ""]
    return lines


def ledger_section(data_root: Path) -> list[str]:
    frame = common.ledger_records(data_root)
    lines = ["## Wall-clock ledger", ""]
    if frame.empty:
        return lines + [
            f"{TO_VERIFY}: no ledger records under `{data_root}/e4_ledger`.",
            "",
        ]
    lines += [
        "| start (UTC) | kind | run | rc | wall (min) | outcome |",
        "|---|---|---|---|---|---|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.start_utc} | {row.kind} | `{row.run_id}` | {row.rc} | "
            f"{row.elapsed_s / 60:.1f} | {row.outcome} |"
        )
    gpu = frame.loc[frame["kind"] == "collect", "elapsed_s"].sum() / 3600
    cpu = frame.loc[frame["kind"] != "collect", "elapsed_s"].sum() / 3600
    lines += [
        "",
        f"Collection (GPU) wall: **{gpu:.2f} h**; probe (CPU) wall: **{cpu:.2f} h**.",
        "",
    ]
    return lines


def flagged_section(rows: list[RunNumbers], labels: tuple[str, str]) -> list[str]:
    lines = ["## Low-competence task flags (R12; kept in every analysis)", ""]
    for row in rows:
        if row.label not in labels or row.source.startswith("missing"):
            continue
        tasks = row.flagged_tasks
        lines.append(
            f"- `{row.run_id}`: {tasks if tasks else 'none'}"
            if tasks is not None
            else f"- `{row.run_id}`: {TO_VERIFY} (no e4_derived/competence.csv)"
        )
    return lines + [""]


def render(plan: dict[str, Any], data_root: Path, existing_root: Optional[Path]) -> str:
    rows = collect_rows(plan, data_root, existing_root)
    labels = (plan["checkpoints"]["sft"]["label"], plan["checkpoints"]["grpo"]["label"])
    lines = [
        "# E4 report — SFT vs GRPO on the unchanged hijack→probe battery",
        "",
        f"Generated {common.utc_now()} by `e4_report.py` from `{data_root}`. "
        "Every number is read from a results file; † = transcribed from "
        "`reports/p0_wave2_report.md` because that run's analysis directory was "
        "not available here. `[TO VERIFY]` = not found, not estimated.",
        "",
        "Rows: the existing wave-2 checkpoint for that suite (goal-traj1 on "
        "libero_goal; that suite's own SFT checkpoint elsewhere), "
        f"{labels[0]} (`{plan['checkpoints']['sft']['hf_repo']}`) and {labels[1]} "
        f"(`{plan['checkpoints']['grpo']['hf_repo']}`). Probe rungs are the best of "
        "27 layer×pool cells ± fold sd, read against the same-family P0 floor.",
        "",
    ]
    lines += readout(plan, rows)
    for suite in SUITES:
        suite_rows = [r for r in rows if r.suite == suite]
        for transform in ("swap", "mirror", "freeze"):
            subset = [r for r in suite_rows if r.transform == transform]
            if not subset:
                continue
            for family in FAMILIES:
                if family == "mlp" and suite not in plan["probes"]["mlp"]["suites"]:
                    continue
                lines += [f"## libero_{suite} / {transform} / {family}", ""]
                lines += table_rows(subset, family) + [""]
    lines += ledger_section(data_root)
    lines += flagged_section(rows, labels)
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", default=str(common.PROBE_DIR / "data"))
    parser.add_argument(
        "--existing-root", default=None, help="wave-2 data with analysis dirs"
    )
    parser.add_argument("--plan", default=str(common.PLAN_PATH))
    parser.add_argument(
        "--out", default=str(common.PROBE_DIR / "reports" / "e4_report.md")
    )
    args = parser.parse_args(argv)
    text = render(
        common.load_plan(Path(args.plan)),
        Path(args.data_root),
        Path(args.existing_root) if args.existing_root else None,
    )
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
