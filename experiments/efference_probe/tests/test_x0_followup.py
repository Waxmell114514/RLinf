"""Offline tests for ``x0_followup.py`` (F1-F7) on a synthetic run."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efference_probe import x0_followup as fu  # noqa: E402
from efference_probe import x0_mechanism as x0  # noqa: E402
from efference_probe.analysis import AnalysisConfig, make_samples  # noqa: E402
from efference_probe.datasets import build_features, load_run  # noqa: E402
from efference_probe.probes import _outer_splits, run_probe  # noqa: E402
from efference_probe.tests.synthetic import make_synthetic_run  # noqa: E402


@pytest.fixture(scope="module")
def run_and_samples(tmp_path_factory):
    run_dir = make_synthetic_run(
        tmp_path_factory.mktemp("fu") / "e4_sft_goal_swap_s1",
        n_tasks=3,
        episodes_per_task=10,
        n_calls=24,
        hidden_signal=0.8,
        command_mean=0.4,
        seed=7,
    )
    run = load_run(run_dir)
    samples = make_samples(run, AnalysisConfig())
    (run_dir / "analysis").mkdir(exist_ok=True)
    samples.to_csv(run_dir / "analysis" / "probe_samples.csv", index=False)
    return run, samples


def test_fold_scorer_reproduces_run_probe_on_true_mismatch(run_and_samples):
    run, samples = run_and_samples
    x, _ = build_features(run, samples, ["mismatch"])
    y = samples["y"].to_numpy(dtype=int)
    groups = samples["episode_id"].to_numpy()
    expected = run_probe(x, y, groups, name="P2r", blocks=["mismatch"])
    folds = [
        fu._linear_score(x[tr], y[tr], groups[tr], x[te], y[te])
        for tr, te in _outer_splits(np.zeros((len(y), 1)), y, groups, 5, 0)
    ]
    np.testing.assert_allclose(folds, expected.fold_balanced_acc, atol=1e-12)


def _f3(checkpoint, outcome, command, p2r=0.8):
    return {
        "checkpoint": checkpoint,
        "true_a+decoded_d": outcome,
        "decoded_a+true_d": command,
        "p2r_csv": p2r,
    }


def test_f3_reading_follows_the_preregistered_rule():
    ok, bad = 0.75, 0.6
    sft = [_f3("Base-Lora", ok, ok)] * 4 + [_f3("Base-Lora", bad, bad)]
    assert fu.f3_reading(sft)["reading"] == "format supports the comparison"
    sft = [_f3("Base-Lora", bad, ok)] * 4 + [_f3("Base-Lora", ok, ok)]
    assert fu.f3_reading(sft)["reading"] == "outcome decode is the bottleneck"
    sft = [_f3("Base-Lora", ok, bad)] * 4 + [_f3("Base-Lora", ok, ok)]
    assert fu.f3_reading(sft)["reading"] == "command decode is the bottleneck"
    sft = [_f3("Base-Lora", bad, bad)] * 5
    grpo = [_f3("GRPO", ok, ok)] * 4
    assert fu.f3_reading(sft + grpo)["reading"] == "both decodes lossy"


def test_f2_reading_follows_the_preregistered_rule():
    def r(all_gain, hijack_gain):
        return {"all": {"gain": all_gain}, "hijack": {"gain": hijack_gain}}

    assert fu.f2_reading([r(0.01, 0.02)] * 4).startswith("no motion trace")
    assert (
        fu.f2_reading([r(0.01, 0.08)] * 3 + [r(0.0, 0.0)])
        == "motion trace beyond state"
    )
    assert fu.f2_reading([r(0.06, 0.01)] * 4) == "mixed"


def test_f2_residual_runs_and_reports_both_subsets(run_and_samples):
    run, _ = run_and_samples
    result = fu.f2_residual(
        run.run_dir.parent, run.run_dir.name, (run.layers[-1], run.pools[0])
    )
    for subset in ("all", "hijack"):
        scores = result[subset]
        assert np.isfinite(scores["gain"])
        assert scores["gain"] == pytest.approx(scores["S2+h"] - scores["S2"])
    assert result["n_hijack"] > 0


def test_matched_support_keeps_overlap_and_drops_disjoint_calls():
    rng = np.random.default_rng(0)
    mine = rng.normal(size=(200, 3))
    tasks = np.zeros(200, dtype=int)
    same = fu.matched_support(mine, tasks, rng.normal(size=(200, 3)), tasks)
    assert 0.3 < same.mean() < 0.8
    far = fu.matched_support(mine + 20.0, tasks, rng.normal(size=(200, 3)), tasks)
    assert not far.any()


def test_margin_shares_sum_to_one():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    predicted = np.array([0, 1, 0, 1, 0, 0, 1, 1])
    tasks = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    frame = fu.margin_shares(y, predicted, tasks)
    assert frame["margin_share"].sum() == pytest.approx(1.0)
    assert frame.set_index("task_id").loc[0, "bacc"] == 1.0


def test_blas_setting_reads_only_reproduced_rows(tmp_path):
    x0_dir = x0.out_dir(tmp_path)
    pd.DataFrame(
        [
            {"run": "a", "blas": "avx2_1t", "reproduced": True},
            {"run": "b", "blas": "avx512_1t", "reproduced": False},
        ]
    ).to_csv(x0_dir / "a7_linear_selected.csv", index=False)
    assert fu.blas_setting(tmp_path, "a") == "avx2_1t"
    assert fu.blas_setting(tmp_path, "b") is None
    assert set(fu.BLAS_ENV) == set(x0.BLAS_PASSES)


def test_f5_reading_reports_overlapping_conditions_as_indeterminate():
    def r(p1, floor, resid):
        return fu.f5_reading({"P1": p1, "floor": floor, "P1_resid": resid})

    assert r(0.60, 0.53, 0.59)["reading"] == "state-orthogonal"
    assert r(0.60, 0.53, 0.54)["reading"] == "state-explained"
    assert r(0.60, 0.53, 0.565)["reading"] == "partial"
    assert r(0.563, 0.528, 0.545)["reading"].startswith("indeterminate")
