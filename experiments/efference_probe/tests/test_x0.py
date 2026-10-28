"""Offline tests for ``x0_mechanism.py`` on a synthetic run.

They pin what the X0 pre-registration requires of the re-implementation: the
comparison features built from arrays equal the probe code's ``mismatch``
block on true inputs, and the out-of-fold refit reproduces ``run_probe``'s
fold scores and selected C exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efference_probe import x0_mechanism as x0  # noqa: E402
from efference_probe.analysis import AnalysisConfig, make_samples  # noqa: E402
from efference_probe.datasets import (  # noqa: E402
    HiddenStore,
    _delta_states,
    build_features,
    call_lookup,
    load_run,
)
from efference_probe.probes import run_probe  # noqa: E402
from efference_probe.tests.synthetic import make_synthetic_run  # noqa: E402


@pytest.fixture(scope="module")
def run_and_samples(tmp_path_factory):
    run_dir = make_synthetic_run(
        tmp_path_factory.mktemp("x0") / "e4_grpo_goal_swap_s1",
        n_tasks=3,
        episodes_per_task=10,
        n_calls=24,
        hidden_signal=0.8,
        command_mean=0.4,
        seed=5,
    )
    run = load_run(run_dir)
    samples = make_samples(run, AnalysisConfig())
    assert len(samples) > 40
    return run, samples


def test_mismatch_from_arrays_equals_the_probe_block(run_and_samples):
    run, samples = run_and_samples
    reference, _ = build_features(run, samples, ["mismatch"])
    prev = list(zip(samples["episode_id"], samples["prev_call"]))
    commanded = x0.states(run, prev, "a_cmd_env").reshape(
        -1, int(run.schema["num_action_chunks"]), int(run.schema["action_dim"])
    )
    delta = _delta_states(run.calls, call_lookup(run.calls), prev)
    np.testing.assert_array_equal(x0.mismatch_from_arrays(commanded, delta), reference)


def test_compose_delta_equals_delta_states(run_and_samples):
    run, samples = run_and_samples
    prev = list(zip(samples["episode_id"], samples["prev_call"]))
    expected = _delta_states(run.calls, call_lookup(run.calls), prev)
    composed = x0.compose_delta(
        x0.states(run, prev, "states_before"), x0.states(run, prev, "states_after")
    )
    np.testing.assert_allclose(composed, expected, atol=1e-6)


@pytest.mark.parametrize("blocks", [["h_cur"], ["a_cmd", "h_cur"], ["dstates"]])
def test_oof_refit_reproduces_run_probe(run_and_samples, blocks):
    run, samples = run_and_samples
    layer, pool = (run.layers[-1], run.pools[0]) if "h_cur" in blocks else (None, None)
    store = HiddenStore(run.run_dir / "hidden")
    x, _ = build_features(run, samples, blocks, layer=layer, pool=pool, store=store)
    y = samples["y"].to_numpy(dtype=int)
    groups = samples["episode_id"].to_numpy()
    expected = run_probe(x, y, groups, name="t", blocks=blocks, layer=layer, pool=pool)
    predicted, folds, chosen = x0.oof_predictions(x, y, groups, "linear")
    np.testing.assert_allclose(folds, expected.fold_balanced_acc, atol=1e-12)
    assert chosen == expected.selected_c
    assert set(np.unique(predicted)) <= {0, 1}


def test_readout_pairs_keep_only_calls_with_a_successor(run_and_samples):
    run, _ = run_and_samples
    pairs = x0.readout_pairs(run, max_calls=50)
    assert len(pairs) == 50
    present = set(zip(run.calls["episode_id"], run.calls["call_idx"]))
    assert all(
        (e, c + 1) in present for e, c in zip(pairs["episode_id"], pairs["call"])
    )


def test_cluster_bootstrap_spearman_resamples_episodes():
    rng = np.random.default_rng(0)
    novelty = rng.normal(size=200)
    frame = pd.DataFrame(
        {
            "run": "r",
            "episode_id": np.repeat(np.arange(40), 5),
            "d_novelty": novelty,
            "d_entropy": novelty + 0.1 * rng.normal(size=200),
        }
    )
    result = x0.cluster_bootstrap_spearman(frame, b=200)
    assert result["n_clusters"] == 40
    assert 0.9 < result["ci_low"] <= result["rho"] <= result["ci_high"] <= 1.0


def test_a2b_scores_hidden_and_state_predictors_by_label(run_and_samples):
    run, samples = run_and_samples
    analysis = run.run_dir / "analysis"
    analysis.mkdir(exist_ok=True)
    samples.to_csv(analysis / "probe_samples.csv", index=False)
    result = x0.a2b_motion_beyond_state(
        run.run_dir.parent, run.run_dir.name, (run.layers[-1], run.pools[0])
    )
    assert result["n_hijack"] > 0
    assert set(result["r2"]) == {"H", "S1", "S2", "H+S1"}
    for scores in result["r2"].values():
        assert set(scores) == {"all", "self", "hijack"}
        assert all(np.isfinite(v) for v in scores.values())
    assert set(result["flags"]) == {
        "regularity explains A2 (ii)",
        "h(c+1) carries motion beyond the state",
    }


def test_cli_runs_a2b_from_the_a2_best_cell(run_and_samples):
    run, samples = run_and_samples
    analysis = run.run_dir / "analysis"
    analysis.mkdir(exist_ok=True)
    samples.to_csv(analysis / "probe_samples.csv", index=False)
    root = run.run_dir.parent
    out = x0.out_dir(root)
    pd.DataFrame(
        [
            {
                "target": "delta_from_next",
                "layer": run.layers[0],
                "pool": run.pools[0],
                "r2_mean": 0.5,
            }
        ]
    ).to_csv(out / f"a2_{run.run_dir.name}.csv", index=False)
    assert x0.main(["--data-root", str(root), "a2b", run.run_dir.name]) == 0
    assert (out / f"a2b_{run.run_dir.name}.json").is_file()
