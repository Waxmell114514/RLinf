"""Offline tests for the wave-3 additions (no GPU, no LIBERO, no checkpoint).

Run with ``pytest experiments/efference_probe/tests/test_wave3.py -q``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efference_probe.compare import (  # noqa: E402
    MIN_EFFECT,
    attribution,
    compare,
    compensation,
    cost_difference,
    load_run_summary,
)

# --------------------------------------------------------------------------
# E2: cross-checkpoint comparison
# --------------------------------------------------------------------------


def _write_fake_run(
    root: Path,
    name: str,
    hidden_score: float,
    floor: float,
    success_probe: float,
    success_clean: float,
    extra_as_markdown: bool = False,
    n_probe: int = 400,
    n_clean: int = 100,
) -> Path:
    run_dir = root / name
    analysis = run_dir / "analysis"
    analysis.mkdir(parents=True)
    rows = []
    for layer in (4, 16):
        for pool in ("ctx_mean", "act_mean"):
            rows.append(("P0", layer, pool, floor - (0.01 if layer == 4 else 0.0)))
            for probe in ("P1", "P3", "P4"):
                bump = 0.0 if (layer, pool) != (16, "act_mean") else 0.01
                rows.append((probe, layer, pool, hidden_score - 0.01 + bump))
    for probe, score in (("P2", 0.5), ("P2r", 0.75), ("C_cmd", 0.5), ("C_phase", 0.5)):
        rows.append((probe, np.nan, np.nan, score))
    table = pd.DataFrame(rows, columns=["name", "layer", "pool", "balanced_acc_mean"])
    table["balanced_acc_std"] = 0.02
    table["n_samples"] = 1200
    table.to_csv(analysis / "probe_results.csv", index=False)

    extra = {
        "selection_floor": {"max_over_cells": floor},
        "permutation_null": {"p95": floor - 0.01},
        "e3": {
            "entropy_mean": {
                "paired_mean_difference": 0.01,
                "wilcoxon_p": 0.02,
                "paired_n_episodes": 150,
            },
            "logprob_sum": {
                "paired_mean_difference": -1.0,
                "wilcoxon_p": 0.03,
                "paired_n_episodes": 150,
            },
        },
        "undo_alignment": {"hijack_mean_alignment": 0.3, "self_mean_alignment": 0.1},
    }
    if extra_as_markdown:
        (analysis / "summary.md").write_text(
            "### Run\n\n#### Probe results\n\n| a |\n\n#### Extra\n\n```json\n"
            + json.dumps(extra, indent=2)
            + "\n```\n",
            encoding="utf-8",
        )
    else:
        (analysis / "extra.json").write_text(json.dumps(extra), encoding="utf-8")

    rng = np.random.default_rng(0)
    episodes = []
    for index in range(n_probe + n_clean):
        is_probe = index < n_probe
        rate = success_probe if is_probe else success_clean
        episodes.append(
            {
                "episode_id": index,
                "success": bool(rng.random() < rate),
                "censored": False,
                "is_probe_episode": is_probe,
            }
        )
    # A censored clean failure must not count against the clean success rate.
    episodes.append(
        {
            "episode_id": 10**6,
            "success": False,
            "censored": True,
            "is_probe_episode": False,
        }
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"episodes": episodes}), encoding="utf-8"
    )
    return run_dir


def test_attribution_margin_against_own_floor(tmp_path):
    run_dir = _write_fake_run(
        tmp_path,
        "rl",
        hidden_score=0.60,
        floor=0.53,
        success_probe=0.9,
        success_clean=0.95,
    )
    result = attribution(load_run_summary("rl", run_dir))
    assert result["floor"] == pytest.approx(0.53)
    assert result["best_hidden"] == pytest.approx(0.60)
    assert result["best_hidden_probe"] in ("P1", "P3", "P4")
    assert result["P1_cell"] == "L16/act_mean"
    assert result["margin"] == pytest.approx(0.07)
    assert result["readable"] is True
    null_dir = _write_fake_run(
        tmp_path,
        "sft",
        hidden_score=0.54,
        floor=0.53,
        success_probe=0.4,
        success_clean=0.5,
    )
    assert attribution(load_run_summary("sft", null_dir))["readable"] is False
    assert MIN_EFFECT == pytest.approx(0.04)


def test_extra_parsed_from_summary_markdown(tmp_path):
    run_dir = _write_fake_run(
        tmp_path,
        "old",
        hidden_score=0.54,
        floor=0.53,
        success_probe=0.4,
        success_clean=0.5,
        extra_as_markdown=True,
    )
    run = load_run_summary("old", run_dir)
    assert run.extra["selection_floor"]["max_over_cells"] == pytest.approx(0.53)
    assert compensation(run)["e3_logprob_sum_p"] == pytest.approx(0.03)


def test_compensation_excludes_censored_and_ci_brackets_point(tmp_path):
    run_dir = _write_fake_run(
        tmp_path,
        "sft",
        hidden_score=0.54,
        floor=0.53,
        success_probe=0.4,
        success_clean=0.6,
    )
    result = compensation(load_run_summary("sft", run_dir), n_bootstrap=2000)
    assert result["n_clean_episodes"] == 100
    low, high = result["hijack_cost_ci95"]
    assert low <= result["hijack_cost"] <= high
    assert result["hijack_cost"] > 0
    assert result["undo_alignment_excess"] == pytest.approx(0.2)


def test_cost_difference_sign_and_grid(tmp_path):
    sft = load_run_summary(
        "sft:none",
        _write_fake_run(
            tmp_path, "sft", 0.54, 0.53, success_probe=0.35, success_clean=0.55
        ),
    )
    rl = load_run_summary(
        "rl:none",
        _write_fake_run(
            tmp_path, "rl", 0.54, 0.53, success_probe=0.93, success_clean=0.95
        ),
    )
    difference = cost_difference(rl, sft, n_bootstrap=2000)
    assert difference["hijack_cost_difference"] < 0  # RL pays less per hijack
    assert difference["p_run_not_lower"] < 0.05
    report = compare([sft, rl], baseline="sft:none", out_dir=tmp_path / "cmp")
    assert (tmp_path / "cmp" / "compare.md").is_file()
    assert "| none |" in report["markdown"]
    assert "not run" not in report["markdown"]
    payload = json.loads((tmp_path / "cmp" / "compare.json").read_text())
    assert payload["versus_baseline"][0]["label"] == "rl:none"
