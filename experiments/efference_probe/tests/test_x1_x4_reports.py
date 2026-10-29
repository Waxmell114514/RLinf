"""X1 and X4 readouts on synthetic data (PREREG_X1_X4.md, the readout rules)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efference_probe import e4_wave as wave_mod  # noqa: E402
from efference_probe import monitor as mon  # noqa: E402
from efference_probe import x1_report as x1  # noqa: E402
from efference_probe import x4_report as x4  # noqa: E402
from efference_probe.analysis import AnalysisConfig, make_samples  # noqa: E402
from efference_probe.datasets import build_features, load_run  # noqa: E402
from efference_probe.probes import run_probe  # noqa: E402
from efference_probe.tests.synthetic import make_synthetic_run  # noqa: E402

PROBE_DIR = Path(__file__).resolve().parents[1]


def _x1_run(root: Path, name: str, seed: int) -> Path:
    """A synthetic run carrying X1's extra observable columns."""
    run_dir = make_synthetic_run(
        root / name, n_tasks=2, episodes_per_task=10, n_calls=24, seed=seed
    )
    calls = pd.read_parquet(run_dir / "calls.parquet")
    rng = np.random.default_rng(seed)
    n = len(calls)
    calls["joint_pos_before"] = list(rng.normal(size=(n, 7)).astype(np.float32))
    calls["joint_pos_after"] = list(rng.normal(size=(n, 7)).astype(np.float32))
    calls["objects_before"] = list(rng.normal(size=(n, 14)).astype(np.float32))
    calls["objects_after"] = list(rng.normal(size=(n, 14)).astype(np.float32))
    calls.to_parquet(run_dir / "calls.parquet", index=False)
    run = load_run(run_dir)
    (run_dir / "analysis").mkdir()
    make_samples(run, AnalysisConfig()).to_csv(
        run_dir / "analysis" / "probe_samples.csv", index=False
    )
    return run_dir


@pytest.fixture(scope="module")
def x1_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("x1data")
    _x1_run(root, "x1_sft_swap", 31)
    _x1_run(root, "x1_grpo_swap", 32)
    return root


# -- X1 -----------------------------------------------------------------------


def test_x1_controls_probe_each_observables_twin(x1_root):
    result = x1.controls(x1_root, "x1_sft_swap")
    assert set(result) == {"run", *x1.CONTROLS}
    n_tasks = 2
    assert result["C_joint"]["n_features"] == 7 + 7 + n_tasks
    assert result["C_scene"]["n_features"] == 14 + 14 + n_tasks
    state = 8 + n_tasks + 8 * n_tasks + 8
    assert result["C_state+d"]["n_features"] == state
    assert result["C_obs"]["n_features"] == state + 28 + 14
    for name in x1.CONTROLS:
        assert 0.0 <= result[name]["bacc"] <= 1.0


def test_x1_r2_needs_every_twin_plus_delta_and_cmd_in_band():
    numbers = {
        "C_dstates": {"bacc": 0.55},
        "C_cmd": {"bacc": 0.50},
        "P1": {"bacc": 0.66},
        "P3": {"bacc": 0.60},
        "P4": {"bacc": 0.70},
    }
    controls = {name: {"bacc": 0.58} for name in x1.CONTROLS}
    controls["C_scene"] = {"bacc": 0.62}
    out = x1.r2_beyond(numbers, controls)
    assert out["bar"] == pytest.approx(0.66)
    assert out["beyond"] == ["P1", "P4"]
    numbers["C_cmd"] = {"bacc": 0.56}
    assert x1.r2_beyond(numbers, controls)["beyond"] == []


def _boot(point, low, high, reproduced=True):
    return {
        "reproduced": reproduced,
        "P3-P1": {"point": point, "ci_low": low, "ci_high": high},
    }


def test_x1_r3_reading_follows_the_rule():
    small = {"x1_sft_swap": 0.01, "x1_grpo_swap": -0.02}
    assert x1.r3_reading(small, _boot(-0.02, -0.05, 0.02), []) == (
        "no efference increment on two pairs"
    )
    big = {"x1_grpo_swap": 0.06}
    assert x1.r3_reading(big, _boot(0.06, 0.02, 0.1), [_boot(0.05, 0.01, 0.09)]) == (
        "RL produces an efference increment"
    )
    assert x1.r3_reading(big, _boot(0.06, 0.02, 0.1), [_boot(0.0, -0.03, 0.03)]) == (
        "unresolved"
    )
    assert x1.r3_reading(small, None, []).startswith("unresolved")
    assert x1.r3_reading(small, _boot(0, -1, 1, reproduced=False), []).startswith(
        "unresolved"
    )


def test_x1_fixed_cell_refit_matches_run_probe_and_differences_bootstrap(x1_root):
    cell = (8, "ctx_mean")
    grpo = x1.fixed_cell(x1_root, "x1_grpo_swap", cell)
    run = load_run(x1_root / "x1_grpo_swap")
    samples = pd.read_csv(x1_root / "x1_grpo_swap" / "analysis" / "probe_samples.csv")
    x, _ = build_features(run, samples, ["h_cur"], layer=8, pool="ctx_mean")
    expected = run_probe(
        x,
        samples["y"].to_numpy(int),
        samples["episode_id"].to_numpy(),
        name="P1",
        blocks=["h_cur"],
        layer=8,
        pool="ctx_mean",
    )
    np.testing.assert_allclose(
        grpo["fold_bacc"], expected.fold_balanced_acc, atol=1e-12
    )
    sft = x1.fixed_cell(x1_root, "x1_sft_swap", cell)
    diff = x1.fixed_cell_difference(grpo, sft)
    assert diff["ci_low"] <= diff["point"] <= diff["ci_high"]


def test_x1_r4_reading_follows_the_rule():
    up = {"point": 0.06, "ci_low": 0.01, "ci_high": 0.1}
    flat = {"point": 0.01, "ci_low": -0.03, "ci_high": 0.05}
    assert x1.r4_reading(up, [up, up]) == "RL raises P1 at the fixed cell"
    assert x1.r4_reading(flat, [up, up]) == "no RL effect at the fixed cell"
    assert x1.r4_reading(up, [up, flat]) == "mixed"


def test_x1_residualised_p1_runs_on_a_probed_run(x1_root):
    run_dir = x1_root / "x1_grpo_swap"
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_DIR / "run_probes.py"),
            "--run",
            str(run_dir),
            "--stage",
            "pilot",
            "--jobs",
            "2",
            "--no-figures",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    out = x1.residualised_p1(x1_root, "x1_grpo_swap")
    assert len(out["P1_resid_folds"]) == 5
    assert set(out["flags"]) == {"state-orthogonal", "state-explained"}


def _manifest(path: Path, episodes: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(json.dumps({"episodes": episodes}))


def test_x1_hijack_cost_is_clean_minus_probe_per_run(tmp_path):
    def episodes(clean, probe):
        rows = [
            {"success": s, "is_probe_episode": False, "censored": False} for s in clean
        ]
        rows += [
            {"success": s, "is_probe_episode": True, "censored": False} for s in probe
        ]
        return rows

    _manifest(tmp_path / "a", episodes([1, 1, 1, 0], [1, 0, 0, 0]))  # cost 0.5
    _manifest(tmp_path / "b", episodes([1, 1], [1, 1, 1, 0]))  # cost 0.25
    out = x1.hijack_cost(tmp_path, "a", "b")
    assert out["sft_cost"] == pytest.approx(0.5)
    assert out["grpo_cost"] == pytest.approx(0.25)
    assert out["rl_minus_sft"] == pytest.approx(-0.25)
    assert out["ci_low"] <= out["rl_minus_sft"] <= out["ci_high"]


# -- X4 -----------------------------------------------------------------------


def _x4_tables(success):
    """Episodes with given success rates per (probe, condition); calls for FPR/TPR."""
    rows = []
    for (probe, condition), (n, rate) in success.items():
        k = int(round(n * rate))
        for i in range(n):
            rows.append(
                {
                    "success": i < k,
                    "is_probe_episode": probe,
                    "monitor_condition": condition,
                    "censored": False,
                    "n_hijacks": 3 if probe else 0,
                }
            )
    rows.append(
        {
            "success": True,
            "is_probe_episode": True,
            "monitor_condition": "off",
            "censored": True,
            "n_hijacks": 3,
        }
    )
    calls = pd.DataFrame(
        {
            "episode_id": ["r:0"] * 4,
            "call_idx": [4, 5, 6, 7],
            "label": ["SELF", "SELF", "HIJACK", "HIJACK"],
            "monitor_condition": ["off"] * 4,
            "monitor_score": [0.1, 2.0, 3.0, 0.2],
            "is_recovery": [False] * 4,
            "states_before": [np.zeros(8)] * 4,
            "states_after": [np.zeros(8)] * 4,
        }
    )
    return pd.DataFrame(rows), calls


def test_x4_readout_applies_rules_m_and_o():
    episodes, calls = _x4_tables(
        {
            (True, "off"): (300, 0.70),
            (True, "monitor"): (300, 0.85),
            (True, "oracle"): (300, 0.90),
            (False, "off"): (80, 0.95),
            (False, "monitor"): (80, 0.90),
            (False, "oracle"): (80, 0.95),
        }
    )
    out = x4.readout("grpo", episodes, calls, threshold=1.0)
    assert out["B_mon"]["point"] == pytest.approx(0.15)
    assert out["B_orc"]["point"] == pytest.approx(0.20)
    assert out["rule_M"] and out["rule_O"]
    assert (
        out["reading"] == "the external comparator closes at least half the hijack cost"
    )
    assert out["online_fpr"] == pytest.approx(0.5) and out[
        "online_tpr"
    ] == pytest.approx(0.5)
    assert out["n_censored"] == 1
    assert out["sanity_flag"] is False  # 0.70 is within 0.10 of 0.730
    assert out["hijacks_per_probe_episode"] == dict.fromkeys(x4.CONDITIONS, 3.0)


def test_x4_readout_separates_detection_from_recovery():
    base = {(False, c): (80, 0.95) for c in ("off", "monitor", "oracle")}
    weak_monitor = {
        **base,
        (True, "off"): (300, 0.50),
        (True, "monitor"): (300, 0.53),
        (True, "oracle"): (300, 0.75),
    }
    episodes, calls = _x4_tables(weak_monitor)
    out = x4.readout("sft", episodes, calls, threshold=1.0)
    assert out["rule_O"] and not out["rule_M"]
    assert out["reading"] == "recovery works; detection limits"
    no_recovery = {**weak_monitor, (True, "oracle"): (300, 0.51)}
    episodes, calls = _x4_tables(no_recovery)
    out = x4.readout("sft", episodes, calls, threshold=1.0)
    assert out["reading"].startswith("undo does not recover the cost")


def test_x4_undo_errors_measure_the_return_to_the_pre_chunk_pose():
    calls = pd.DataFrame(
        {
            "episode_id": [0, 0, 0],
            "call_idx": [4, 5, 6],
            "label": ["HIJACK", "RECOVERY", "SELF"],
            "is_recovery": [False, True, False],
            "states_before": [np.r_[0.10, 0.20, 0.30, np.zeros(5)]] * 3,
            "states_after": [
                np.r_[0.20, 0.20, 0.30, np.zeros(5)],
                np.r_[0.11, 0.20, 0.30, np.zeros(5)],  # 1 cm from where call 4 began
                np.r_[0.11, 0.20, 0.30, np.zeros(5)],
            ],
        }
    )
    np.testing.assert_allclose(x4.undo_errors_cm(calls), [1.0], atol=1e-9)


# -- the driver's X1/X4 checks on a real (fake-env) harness run ------------------


@pytest.mark.parametrize("kind", ["x1", "x4"])
def test_extension_checks_pass_on_harness_output(tmp_path, kind):
    pytest.importorskip("torch")
    pytest.importorskip("omegaconf")
    from efference_probe.tests import test_x1_x4_harness as fake

    if kind == "x1":
        overrides = ["capture.extra_observables=true"]
        monitor = None
    else:
        artifact = "experiments/efference_probe/x4/monitor_grpo.json"
        digest = json.loads((PROBE_DIR / "x4" / "monitor_grpo.json").read_text())[
            "sha256"
        ]
        overrides = [
            "monitor.enabled=true",
            f"monitor.artifact={artifact}",
            f"monitor.artifact_sha256={digest}",
            "monitor.assignment=cycle",
        ]
        monitor = mon.LinearMonitor.load(PROBE_DIR / "x4" / "monitor_grpo.json")
    harness = fake._harness(tmp_path, overrides, monitor=monitor)
    harness.writer.write_run_config(harness._run_config_payload())
    calls, _ = fake._collect(harness)
    checks = wave_mod.extension_checks(harness.out_dir, calls, smoke=True)
    assert checks and all(checks.values()), checks


def test_x4_undo_errors_leave_out_undos_of_self_commanded_chunks():
    """Review R4: a false-positive undo has no hijacked call to return to."""
    calls = pd.DataFrame(
        {
            "episode_id": [0, 0],
            "call_idx": [4, 5],
            "label": ["SELF", "RECOVERY"],
            "is_recovery": [False, True],
            "states_before": [np.zeros(8)] * 2,
            "states_after": [np.ones(8)] * 2,
        }
    )
    assert len(x4.undo_errors_cm(calls)) == 0


def test_vision_check_needs_vis_for_every_call(tmp_path):
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    np.savez(hidden / "ep00000.npz", h=np.zeros((3, 1, 1, 4)), vis=np.zeros((3, 8)))
    assert wave_mod._vision_ok(tmp_path)
    np.savez(hidden / "ep00001.npz", h=np.zeros((2, 1, 1, 4)))
    assert not wave_mod._vision_ok(tmp_path)


def test_checkpoint_labels_follow_the_run_id():
    from efference_probe import x0_mechanism as x0

    assert x0.checkpoint("e4_grpo_goal_swap_s1") == "GRPO"
    assert x0.checkpoint("e4_sft_goal_swap_s1") == "Base-Lora"
    assert x0.checkpoint("x1_grpo_swap") == "GRPO-goal"
    assert x0.checkpoint("x1_sft_swap") == "goal-traj1"
    assert x0.checkpoint("x4_grpo_mirror_s3") == "GRPO"


def test_x1_r3_stays_unresolved_until_every_planned_run_is_in():
    small = {"x1_sft_swap": 0.01, "x1_grpo_swap": -0.02}
    reading = x1.r3_reading(small, _boot(-0.02, -0.05, 0.02), [], n_planned=6)
    assert reading == "unresolved (2 of 6 X1 runs)"


def test_main02_reproduction_compares_the_four_rungs():
    from efference_probe import e4_report as report

    wave2 = report.read_wave2_report("main02", "goal-traj1", "goal", "swap", 1)
    ours = report.RunNumbers("goal-traj1", "x1_sft_swap", "goal", "swap", 1, "analysis")
    ours.families["linear"] = {
        r: {"bacc": wave2.families["linear"][r]["bacc"] + 0.01}
        for r in ("P1", "P3", "P4", "P2r")
    }
    out = x1.main02_reproduction(ours)
    assert out["within_0.04"] and out["differences"]["P1"] == pytest.approx(0.01)
    ours.families["linear"]["P2r"]["bacc"] += 0.05
    assert not x1.main02_reproduction(ours)["within_0.04"]
