"""Cross-checkpoint comparison: attribution and compensation, kept apart.

Wave 3's E2 asks whether RL post-training builds the attribution substrate
SFT lacks (Ye's prediction: no -- compensation rises, encoding does not).  The
two halves of that claim are measured by different instruments and must be
reported separately, so this module produces two tables per run:

* **Attribution** -- the probe ladder against the run's *own*
  selection-aware floor and permutation null.  Floors depend on sample size,
  so a raw balanced accuracy is never compared across runs; the margin above
  each run's floor is.
* **Compensation** -- what the policy does about a hijack: the randomised
  success-rate cost of hijacking (clean minus probe episodes), the E3 reaction
  on the next call, and the innovation-corrected undo alignment.

Everything is read from finished ``run_probes.py`` outputs plus the run's
``manifest.json``; nothing is refitted.  Runs labelled ``objective:interface``
(e.g. ``sft:none``, ``rl:none``, ``sft:cmd``) are also laid out as the
interface x objective grid that E2 and E5 fill together.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# -- pre-registered (PREREG_wave3.md, E2) ----------------------------------
# Same yardstick as the workshop paper: the smallest attribution effect the
# n~1.7k design can see is ~0.04 balanced accuracy above the floor.
MIN_EFFECT = 0.04
HIDDEN_PROBES = ("P1", "P3", "P4")
N_BOOTSTRAP = 5000


@dataclass
class RunSummary:
    label: str
    run_dir: Path
    probes: pd.DataFrame
    extra: dict[str, Any]
    episodes: pd.DataFrame
    run_config: dict[str, Any]


def _extra_from_summary_md(text: str) -> dict[str, Any]:
    """Recover the JSON block that ``analysis.summarize`` writes under Extra."""
    marker = text.find("#### Extra")
    if marker < 0:
        return {}
    match = re.search(r"```json\n(.*?)\n```", text[marker:], flags=re.DOTALL)
    return json.loads(match.group(1)) if match else {}


def load_run_summary(
    label: str, run_dir: str | Path, analysis: str = "analysis"
) -> RunSummary:
    run_dir = Path(run_dir)
    analysis_dir = run_dir / analysis
    probes_path = analysis_dir / "probe_results.csv"
    if not probes_path.is_file():
        raise FileNotFoundError(
            f"{probes_path} missing: run run_probes.py --stage main on {run_dir} first"
        )
    probes = pd.read_csv(probes_path)
    extra_path = analysis_dir / "extra.json"
    if extra_path.is_file():
        extra = json.loads(extra_path.read_text(encoding="utf-8"))
    else:
        summary_md = analysis_dir / "summary.md"
        extra = (
            _extra_from_summary_md(summary_md.read_text(encoding="utf-8"))
            if summary_md.is_file()
            else {}
        )
    manifest_path = run_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    episodes = pd.DataFrame(manifest.get("episodes", []))
    run_config: dict[str, Any] = {}
    config_path = run_dir / "run_config.yaml"
    if config_path.is_file():
        import yaml

        with open(config_path, encoding="utf-8") as handle:
            run_config = yaml.safe_load(handle) or {}
    return RunSummary(label, run_dir, probes, extra, episodes, run_config)


# -- attribution -----------------------------------------------------------


def _best(probes: pd.DataFrame, name: str) -> tuple[float, float, str]:
    rows = probes[probes["name"] == name]
    if rows.empty:
        return float("nan"), float("nan"), ""
    best = rows.loc[rows["balanced_acc_mean"].idxmax()]
    cell = ""
    if pd.notna(best.get("layer")) and isinstance(best.get("pool"), str):
        cell = f"L{int(best['layer'])}/{best['pool']}"
    return float(best["balanced_acc_mean"]), float(best["balanced_acc_std"]), cell


def attribution(run: RunSummary) -> dict[str, Any]:
    """Ladder margins against the run's own nulls, plus the verdict."""
    out: dict[str, Any] = {"label": run.label}
    floor = (run.extra.get("selection_floor") or {}).get("max_over_cells")
    if floor is None:
        floor = _best(run.probes, "P0")[0]
    out["floor"] = float(floor)
    out["perm_p95"] = float(
        (run.extra.get("permutation_null") or {}).get("p95", float("nan"))
    )
    hidden_best, hidden_name = -np.inf, ""
    for name in HIDDEN_PROBES + ("P2", "P2r", "C_cmd", "C_dstates", "C_phase"):
        score, sd, cell = _best(run.probes, name)
        out[name] = score
        out[f"{name}_sd"] = sd
        if cell:
            out[f"{name}_cell"] = cell
        if name in HIDDEN_PROBES and np.isfinite(score) and score > hidden_best:
            hidden_best, hidden_name = score, name
    out["best_hidden"] = (
        float(hidden_best) if np.isfinite(hidden_best) else float("nan")
    )
    out["best_hidden_probe"] = hidden_name
    out["margin"] = out["best_hidden"] - out["floor"]
    out["control_margin"] = out["P2r"] - out["floor"]
    above_perm = (
        not np.isfinite(out["perm_p95"]) or out["best_hidden"] > out["perm_p95"]
    )
    out["readable"] = bool(out["margin"] >= MIN_EFFECT and above_perm)
    samples = run.probes[run.probes["name"] == "P1"]["n_samples"]
    out["n_samples"] = int(samples.iloc[0]) if not samples.empty else 0
    return out


# -- compensation ----------------------------------------------------------


def _episode_outcomes(episodes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Success indicators for probe and clean episodes; censored ones dropped."""
    if episodes.empty or "is_probe_episode" not in episodes:
        return np.zeros(0), np.zeros(0)
    frame = episodes
    if "censored" in frame:
        frame = frame[~frame["censored"].astype(bool)]
    probe = frame[frame["is_probe_episode"].astype(bool)]["success"].astype(float)
    clean = frame[~frame["is_probe_episode"].astype(bool)]["success"].astype(float)
    return probe.to_numpy(), clean.to_numpy()


def _bootstrap_cost(
    probe: np.ndarray, clean: np.ndarray, rng: np.random.Generator, n: int
) -> np.ndarray:
    """Bootstrap draws of clean-minus-probe success, resampled per group."""
    if len(probe) == 0 or len(clean) == 0:
        return np.full(n, np.nan)
    probe_draws = rng.choice(probe, size=(n, len(probe)), replace=True).mean(axis=1)
    clean_draws = rng.choice(clean, size=(n, len(clean)), replace=True).mean(axis=1)
    return clean_draws - probe_draws


def compensation(
    run: RunSummary, seed: int = 0, n_bootstrap: int = N_BOOTSTRAP
) -> dict[str, Any]:
    """Behavioural response to hijacks, from logged data only."""
    probe, clean = _episode_outcomes(run.episodes)
    out: dict[str, Any] = {
        "label": run.label,
        "n_probe_episodes": int(len(probe)),
        "n_clean_episodes": int(len(clean)),
        "success_probe": float(probe.mean()) if len(probe) else float("nan"),
        "success_clean": float(clean.mean()) if len(clean) else float("nan"),
    }
    out["hijack_cost"] = out["success_clean"] - out["success_probe"]
    draws = _bootstrap_cost(probe, clean, np.random.default_rng(seed), n_bootstrap)
    out["hijack_cost_ci95"] = (
        [float(np.nanquantile(draws, 0.025)), float(np.nanquantile(draws, 0.975))]
        if np.isfinite(draws).any()
        else [float("nan"), float("nan")]
    )
    if len(probe) and len(clean):
        try:
            from scipy.stats import fisher_exact

            table = [
                [int(probe.sum()), int(len(probe) - probe.sum())],
                [int(clean.sum()), int(len(clean) - clean.sum())],
            ]
            out["hijack_cost_fisher_p"] = float(fisher_exact(table)[1])
        except ImportError:  # pragma: no cover - scipy ships with sklearn
            pass

    e3 = run.extra.get("e3") or {}
    for metric in ("entropy_mean", "logprob_sum"):
        block = e3.get(metric) or {}
        out[f"e3_{metric}_diff"] = float(
            block.get("paired_mean_difference", block.get("difference", float("nan")))
        )
        out[f"e3_{metric}_p"] = float(block.get("wilcoxon_p", float("nan")))
        out[f"e3_{metric}_n_episodes"] = int(block.get("paired_n_episodes", 0) or 0)
    undo = run.extra.get("undo_alignment") or {}
    hijack_alignment = float(undo.get("hijack_mean_alignment", float("nan")))
    self_alignment = float(undo.get("self_mean_alignment", float("nan")))
    out["undo_alignment_hijack"] = hijack_alignment
    out["undo_alignment_self"] = self_alignment
    out["undo_alignment_excess"] = hijack_alignment - self_alignment
    return out


def cost_difference(
    run: RunSummary,
    baseline: RunSummary,
    seed: int = 0,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict[str, Any]:
    """Does ``run`` absorb hijacks better than ``baseline``?  (independent runs)"""
    rng = np.random.default_rng(seed)
    run_probe_ep, run_clean_ep = _episode_outcomes(run.episodes)
    base_probe_ep, base_clean_ep = _episode_outcomes(baseline.episodes)
    a = _bootstrap_cost(run_probe_ep, run_clean_ep, rng, n_bootstrap)
    b = _bootstrap_cost(base_probe_ep, base_clean_ep, rng, n_bootstrap)
    difference = a - b
    if not np.isfinite(difference).any():
        return {"note": "missing episode outcomes"}
    point = (run_clean_ep.mean() - run_probe_ep.mean()) - (
        base_clean_ep.mean() - base_probe_ep.mean()
    )
    return {
        "label": run.label,
        "baseline": baseline.label,
        "hijack_cost_difference": float(point),
        "ci95": [
            float(np.nanquantile(difference, 0.025)),
            float(np.nanquantile(difference, 0.975)),
        ],
        # One-sided: fraction of draws where the run pays *more* than baseline.
        "p_run_not_lower": float(np.nanmean(difference >= 0)),
    }


# -- rendering -------------------------------------------------------------


def _markdown(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    present = [c for c in columns if c in frame.columns]
    lines = ["| " + " | ".join(present) + " |", "|" + "---|" * len(present)]
    for _, row in frame.iterrows():
        cells = []
        for column in present:
            value = row[column]
            if isinstance(value, float):
                cells.append("-" if not np.isfinite(value) else f"{value:.3f}")
            elif isinstance(value, list):
                cells.append("[" + ", ".join(f"{v:.3f}" for v in value) + "]")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _grid(attr: pd.DataFrame, comp: pd.DataFrame) -> list[str]:
    """Interface x objective layout for labels of the form objective:interface."""
    labelled = [label for label in attr["label"] if ":" in label]
    if not labelled:
        return []
    cells: dict[tuple[str, str], str] = {}
    objectives, interfaces = [], []
    for label in labelled:
        objective, interface = label.split(":", 1)
        if objective not in objectives:
            objectives.append(objective)
        if interface not in interfaces:
            interfaces.append(interface)
        a = attr[attr["label"] == label].iloc[0]
        c = comp[comp["label"] == label].iloc[0]
        cells[(interface, objective)] = (
            f"margin {a['margin']:+.3f} ({'readable' if a['readable'] else 'not readable'}); "
            f"hijack cost {c['hijack_cost']:+.3f}"
        )
    lines = ["| interface \\ objective | " + " | ".join(objectives) + " |"]
    lines.append("|" + "---|" * (len(objectives) + 1))
    for interface in interfaces:
        row = [cells.get((interface, objective), "not run") for objective in objectives]
        lines.append(f"| {interface} | " + " | ".join(row) + " |")
    return lines


def compare(
    runs: list[RunSummary],
    baseline: Optional[str] = None,
    out_dir: Optional[Path] = None,
    seed: int = 0,
) -> dict[str, Any]:
    attr = pd.DataFrame([attribution(run) for run in runs])
    comp = pd.DataFrame([compensation(run, seed) for run in runs])
    report: dict[str, Any] = {
        "attribution": attr.to_dict(orient="records"),
        "compensation": comp.to_dict(orient="records"),
        "min_effect": MIN_EFFECT,
    }
    base = next((r for r in runs if r.label == baseline), None) if baseline else None
    differences = []
    if base is not None:
        for run in runs:
            if run.label == base.label:
                continue
            entry = cost_difference(run, base, seed)
            a = attr[attr["label"] == run.label].iloc[0]
            b = attr[attr["label"] == base.label].iloc[0]
            entry["margin_difference"] = float(a["margin"] - b["margin"])
            differences.append(entry)
    report["versus_baseline"] = differences

    lines = ["# Cross-checkpoint comparison", ""]
    lines.append(
        "Attribution and compensation are separate instruments; read them "
        "separately. Balanced accuracies are each run's best cell; `margin` is "
        "that minus the run's own selection-aware floor."
    )
    lines += ["", "## Attribution (probe ladder vs each run's own floor)", ""]
    lines += _markdown(
        attr,
        [
            "label",
            "n_samples",
            "floor",
            "perm_p95",
            "P1",
            "P3",
            "P4",
            "best_hidden",
            "margin",
            "readable",
            "P2",
            "P2r",
            "control_margin",
            "C_cmd",
            "C_dstates",
            "C_phase",
        ],
    )
    lines += [
        "",
        f"`readable`: margin ≥ {MIN_EFFECT} and best hidden above the "
        "permutation p95 (when one was run).",
        "",
        "## Compensation (behaviour after a hijack)",
        "",
    ]
    lines += _markdown(
        comp,
        [
            "label",
            "n_probe_episodes",
            "n_clean_episodes",
            "success_clean",
            "success_probe",
            "hijack_cost",
            "hijack_cost_ci95",
            "hijack_cost_fisher_p",
            "e3_entropy_mean_diff",
            "e3_entropy_mean_p",
            "e3_logprob_sum_diff",
            "e3_logprob_sum_p",
            "undo_alignment_excess",
        ],
    )
    lines += [
        "",
        "`hijack_cost` = success(clean episodes) − success(probe episodes); the "
        "episode type is randomised, so this is the unconfounded contrast. "
        "Lower cost = more compensation. A near-ceiling policy compresses this "
        "number; read it with `success_clean`.",
    ]
    if differences:
        lines += ["", f"## Versus baseline `{baseline}`", ""]
        lines += _markdown(
            pd.DataFrame(differences),
            [
                "label",
                "margin_difference",
                "hijack_cost_difference",
                "ci95",
                "p_run_not_lower",
            ],
        )
    grid = _grid(attr, comp)
    if grid:
        lines += ["", "## Interface × objective", ""] + grid
    report["markdown"] = "\n".join(lines) + "\n"
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "compare.md").write_text(report["markdown"], encoding="utf-8")
        payload = {k: v for k, v in report.items() if k != "markdown"}
        (out_dir / "compare.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
        )
    return report


__all__ = [
    "MIN_EFFECT",
    "RunSummary",
    "attribution",
    "compare",
    "compensation",
    "cost_difference",
    "load_run_summary",
]
