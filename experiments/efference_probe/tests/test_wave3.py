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


# --------------------------------------------------------------------------
# Harness: substep frames
# --------------------------------------------------------------------------


def test_substeps_config_validation():
    from efference_probe.config import RunConfig

    cfg = RunConfig()
    cfg.frames.substeps = True
    with pytest.raises(ValueError, match="skip_intermediate_renders"):
        cfg.validate()
    cfg.env.skip_intermediate_renders = False
    cfg.validate()
    cfg.frames.mode = "none"
    with pytest.raises(ValueError, match="no effect"):
        cfg.validate()


def test_substep_frames_written_with_expected_names(tmp_path):
    from efference_probe.storage import (
        DiskBudget,
        RunWriter,
        call_frame_name,
        substep_frame_name,
    )

    writer = RunWriter(tmp_path / "run", DiskBudget(tmp_path / "run", 1.0, 1.0))
    images = [np.full((8, 8, 3), k * 20, dtype=np.uint8) for k in range(8)]
    writer.save_substep_frames(3, 5, images)
    directory = tmp_path / "run" / "frames" / "ep00003"
    assert sorted(p.name for p in directory.iterdir()) == sorted(
        substep_frame_name(5, k) for k in range(8)
    )
    assert call_frame_name(5) == "call0005_main.jpg"
    assert substep_frame_name(5, 7) == "call0005_s7_main.jpg"


# --------------------------------------------------------------------------
# E3/E4: world-model queries and scores
# --------------------------------------------------------------------------


def _smooth_scene(rng: np.random.Generator) -> np.ndarray:
    coarse = rng.integers(0, 256, size=(16, 16, 3)).astype(np.float32)
    return np.kron(coarse, np.ones((16, 16, 1))).astype(np.uint8)


def _toy_world_run(tmp_path: Path, transform: str) -> Path:
    """A synthetic run whose frames obey the toy WM's own dynamics."""
    import yaml
    from PIL import Image

    from efference_probe.datasets import load_run
    from efference_probe.storage import call_frame_name, substep_frame_name
    from efference_probe.tests.synthetic import make_synthetic_run
    from efference_probe.wm_model import ToyWorldModel

    run_dir = make_synthetic_run(
        tmp_path / transform, transform=transform, command_mean=0.5, n_calls=16, seed=1
    )
    config_path = run_dir / "run_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["config"]["frames"] = {"substeps": True, "mode": "all"}
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    run = load_run(run_dir)
    world = ToyWorldModel(noise=0.0)
    rng = np.random.default_rng(0)
    n_chunks, action_dim = run.schema["num_action_chunks"], run.schema["action_dim"]
    for episode_id, episode in run.calls.groupby("episode_id"):
        directory = run_dir / "frames" / f"ep{int(episode_id):05d}"
        directory.mkdir(parents=True)
        scene = _smooth_scene(rng)
        for _, row in episode.sort_values("call_idx").iterrows():
            call = int(row["call_idx"])
            Image.fromarray(scene).save(directory / call_frame_name(call), quality=95)
            executed = np.asarray(row["a_exec"], dtype=np.float32).reshape(
                n_chunks, action_dim
            )
            substeps = world.predict(scene, [scene] * 4, None, executed)
            for k, image in enumerate(substeps):
                Image.fromarray(image).save(
                    directory / substep_frame_name(call, k), quality=95
                )
            scene = substeps[-1]
    return run_dir


def test_condition_actions_follow_wanenv():
    from efference_probe.wm_data import condition_actions

    reset = condition_actions(None, "substeps")
    assert np.allclose(reset[:, :6], 0) and np.allclose(reset[:, 6], -1)
    previous = np.arange(8 * 7, dtype=np.float32).reshape(8, 7)
    faithful = condition_actions(previous, "substeps")
    assert np.allclose(faithful[0, :6], 0) and faithful[0, 6] == -1
    assert np.allclose(faithful[1:], previous[4:])
    held = condition_actions(previous, "static")
    assert np.allclose(held[1:, :6], 0) and np.allclose(held[1:, 6], previous[-1, 6])


def test_alternatives_exclude_self_and_swap_donor(tmp_path):
    from efference_probe.datasets import load_run
    from efference_probe.wm_data import ChunkTable, FrameStore, build_query

    run_dir = _toy_world_run(tmp_path, "swap")
    run = load_run(run_dir)
    table, frames = ChunkTable(run), FrameStore(run_dir)
    hijacked = run.calls[(run.calls["label"] == "HIJACK") & (run.calls["call_idx"] > 0)]
    row = hijacked.iloc[0]
    query = build_query(
        table,
        frames,
        int(row["episode_id"]),
        int(row["call_idx"]),
        "substeps",
        n_alternatives=4,
        rng=np.random.default_rng(0),
    )
    used = {episode for episode, _ in query.meta["donors"]}
    assert int(row["episode_id"]) not in used
    assert int(row["donor_episode_id"]) not in used
    assert "exec" in query.candidates  # swap: executed differs from commanded
    assert query.actual.shape[0] == 8 and query.context.shape[0] == 4
    assert len(query.alternatives) == 4


def test_orthonormal_basis_handles_degenerate_directions():
    from efference_probe.wm_scores import orthonormal_basis

    zero = np.zeros((4, 4, 3))
    assert orthonormal_basis([zero, zero]).shape[1] == 0
    a = np.random.default_rng(0).normal(size=(4, 4, 3))
    basis = orthonormal_basis([a, 2 * a, zero])
    assert basis.shape[1] == 1


def test_e3_gate_separates_sensitive_from_blind_world_model(tmp_path):
    from efference_probe.datasets import load_run
    from efference_probe.wm_data import ChunkTable, FrameStore, build_query
    from efference_probe.wm_model import ToyWorldModel
    from efference_probe.wm_scores import e3_summary, prepare, sensitivity_row

    run_dir = _toy_world_run(tmp_path, "swap")
    run = load_run(run_dir)
    table, frames = ChunkTable(run), FrameStore(run_dir)
    calls = run.calls[(run.calls["label"] == "SELF") & (run.calls["call_idx"] >= 1)]
    results = {}
    for name, model in (
        ("sensitive", ToyWorldModel()),
        ("blind", ToyWorldModel(blind=True)),
    ):
        rows = []
        for _, row in calls.head(25).iterrows():
            query = build_query(
                table,
                frames,
                int(row["episode_id"]),
                int(row["call_idx"]),
                "substeps",
                n_alternatives=3,
                rng=np.random.default_rng(1),
                with_transforms=True,
            )
            predictions = model.predict_candidates(
                query.reference,
                query.context,
                query.condition,
                query.candidates,
                seed=0,
                extra_seed=1,
            )
            prepared = {k: prepare(v) for k, v in predictions.items()}
            rows.append(
                sensitivity_row(
                    prepared, prepare(query.context[-1]), prepare(query.actual)
                )
            )
        results[name] = e3_summary(pd.DataFrame(rows), "substeps")
    assert results["sensitive"]["r_freeze"]["median"] > 1
    assert results["sensitive"]["fidelity_cmd_beats_alt_median"] > 0.8
    assert results["blind"]["r_freeze"]["median"] < 1e-6
    # 25 queries < the pre-registered 200: reported, never a gate decision.
    assert results["sensitive"]["gate"]["eligible"] is False
    assert results["sensitive"]["gate"]["passed"] is None


def test_e4_cli_end_to_end_on_toy_world(tmp_path):
    from efference_probe import run_wm

    run_dir = _toy_world_run(tmp_path, "freeze")
    out = tmp_path / "e4"
    code = run_wm.main(
        [
            "comparator",
            "--run",
            str(run_dir),
            "--out",
            str(out),
            "--backend",
            "toy",
            "--alternatives",
            "4",
            "--bootstrap",
            "100",
        ]
    )
    assert code == 0
    summary = json.loads((out / "summary.json").read_text())
    freeze = summary["per_transform"]["freeze"]
    assert freeze["n_positive"] > 5
    # The world obeys the toy model, so the command predicts SELF outcomes and
    # misses frozen ones; an index slip anywhere would pull this to ~0.5.
    assert freeze["raw_error"]["auroc"] > 0.9
    assert freeze["adv_distance"]["auroc"] > 0.9
    # Rerunning resumes from rows.csv instead of rescoring.
    rows_before = len(pd.read_csv(out / "rows.csv"))
    run_wm.main(
        [
            "comparator",
            "--run",
            str(run_dir),
            "--out",
            str(out),
            "--backend",
            "toy",
            "--alternatives",
            "4",
            "--bootstrap",
            "50",
        ]
    )
    assert len(pd.read_csv(out / "rows.csv")) == rows_before


def test_e4_verdicts():
    from efference_probe.wm_scores import e4_verdict

    good = {
        "adv_distance": {"auroc": 0.8, "ci95": [0.7, 0.9]},
        "novelty_adv": {"auroc": 0.55, "ci95": [0.45, 0.65]},
        "adv_distance-novelty_adv": {"difference": 0.25, "ci95": [0.1, 0.4]},
    }
    assert e4_verdict(good) == "comparator supplies what the policy lacks"
    novelty_only = {
        "adv_distance": {"auroc": 0.56, "ci95": [0.48, 0.64]},
        "novelty_adv": {"auroc": 0.55, "ci95": [0.47, 0.63]},
        "adv_distance-novelty_adv": {"difference": 0.01, "ci95": [-0.05, 0.07]},
    }
    assert e4_verdict(novelty_only) == "comparator only reads novelty"
