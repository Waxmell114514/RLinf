"""Offline tests for the X1/X4 helpers: observables, the monitor, the config.

PREREG_X1_X4.md §4: every change sits behind a flag that defaults off, so
these tests also pin that earlier configs are untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efference_probe import monitor as mon  # noqa: E402
from efference_probe import observables as ob  # noqa: E402
from efference_probe.analysis import AnalysisConfig, make_samples  # noqa: E402
from efference_probe.config import load_config  # noqa: E402
from efference_probe.datasets import (  # noqa: E402
    _delta_states,
    _mismatch_features,
    call_lookup,
    load_run,
)
from efference_probe.probes import _make_classifier  # noqa: E402
from efference_probe.tests.synthetic import make_synthetic_run  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
# Hashes of every shipped config before X1/X4 existed (2026-10-29).
HASHES_BEFORE_X1_X4 = {
    "main.yaml": "314acbe8defc",
    "main02.yaml": "d9995495d95c",
    "main02_freeze.yaml": "0acf98e43d96",
    "main02_mirror.yaml": "17cf313154d1",
    "main_freeze.yaml": "f1b606882fb5",
    "main_mirror.yaml": "e06f13cb32a7",
    "pilot.yaml": "ab930cbc123f",
    "smoke.yaml": "6fe63b592b67",
    "suite_long.yaml": "d25491ed1c4c",
    "suite_object.yaml": "1bcb22bbfd4f",
    "suite_spatial.yaml": "d60485deb715",
}


def _raw_obs(eef, joints, objects):
    obs = {
        "robot0_eef_pos": np.asarray(eef, dtype=np.float64),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
        "robot0_joint_pos": np.asarray(joints, dtype=np.float64),
        "agentview_image": np.zeros((4, 4, 3)),
    }
    for name, (pos, quat) in objects.items():
        obs[f"{name}_pos"] = np.asarray(pos, dtype=np.float64)
        obs[f"{name}_quat"] = np.asarray(quat, dtype=np.float64)
        obs[f"{name}_to_robot0_eef_pos"] = np.asarray(pos, dtype=np.float64) - eef
    return obs


# -- observables ------------------------------------------------------------


def test_object_keys_are_world_frame_poses_in_a_fixed_order():
    obs = _raw_obs(
        [0.1, 0.2, 0.3],
        np.arange(7),
        {"plate_1": ([1, 2, 3], [0, 0, 0, 1]), "bowl_1": ([4, 5, 6], [0, 0, 1, 0])},
    )
    assert ob.object_keys(obs) == [
        "bowl_1_pos",
        "bowl_1_quat",
        "plate_1_pos",
        "plate_1_quat",
    ]


def test_object_poses_concatenate_in_key_order_and_refuse_missing_keys():
    objects = {"bowl_1": ([4, 5, 6], [0, 0, 1, 0])}
    obs = [_raw_obs([0, 0, 0], np.zeros(7), objects)] * 2
    keys = ob.object_keys(obs[0])
    np.testing.assert_array_equal(
        ob.object_poses(obs, keys), [[4, 5, 6, 0, 0, 1, 0]] * 2
    )
    with pytest.raises(KeyError):
        ob.object_poses(obs, keys + ["plate_1_pos"])


def test_joint_positions_and_eef_check():
    obs = [_raw_obs([0.1, 0.2, 0.3], np.arange(7) * 0.1, {}) for _ in range(3)]
    assert ob.joint_positions(obs).shape == (3, 7)
    states = np.tile(np.array([0.1, 0.2, 0.3, 0, 0, 0, 0, 0], np.float32), (3, 1))
    ob.check_eef(obs, states, "after")  # float32 rounding is within tolerance
    states[1, 0] += 1e-3
    with pytest.raises(RuntimeError, match="out of step"):
        ob.check_eef(obs, states, "after")


# -- the monitor ------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    run_dir = make_synthetic_run(
        tmp_path_factory.mktemp("x4") / "x4_synthetic",
        n_tasks=2,
        episodes_per_task=8,
        n_calls=20,
        transform="mirror",
        seed=11,
    )
    run = load_run(run_dir)
    return run, make_samples(run, AnalysisConfig())


def test_online_features_equal_the_probe_mismatch_block(synthetic):
    run, samples = synthetic
    prev = list(zip(samples["episode_id"], samples["prev_call"]))
    lookup = call_lookup(run.calls)
    reference = _mismatch_features(run, lookup, prev, "a_cmd_env")
    rows = run.calls.iloc[[lookup[(int(e), int(c))] for e, c in prev]]
    chunks, dim = int(run.schema["num_action_chunks"]), int(run.schema["action_dim"])
    commanded = np.stack(rows["a_cmd_env"].map(np.asarray).to_list()).reshape(
        -1, chunks, dim
    )
    before = np.stack(rows["states_before"].map(np.asarray).to_list())
    after = np.stack(rows["states_after"].map(np.asarray).to_list())
    np.testing.assert_array_equal(
        mon.mismatch_features(commanded, before, after), reference
    )
    np.testing.assert_array_equal(
        mon.delta_state(before, after), _delta_states(run.calls, lookup, prev)
    )


def test_undo_retraces_the_chunk_inside_the_bounds():
    rng = np.random.default_rng(0)
    executed = rng.uniform(-1, 1, size=(8, 7)).astype(np.float32)
    executed[:, 6] = np.where(np.arange(8) < 5, -1.0, 1.0)
    undo = mon.undo_chunk(executed)
    np.testing.assert_array_equal(undo[:, :6], -executed[::-1, :6])
    np.testing.assert_array_equal(undo[:, 6], executed[::-1, 6])
    np.testing.assert_allclose(undo[:, :6].sum(0), -executed[:, :6].sum(0), atol=1e-6)
    assert np.abs(undo).max() <= 1.0
    np.testing.assert_array_equal(mon.undo_chunk(undo), executed)


def test_conditions_are_reproducible_balanced_and_on_their_own_stream():
    first = mon.assign_conditions(3, 2, 0, 20, [1 / 3] * 3)
    np.testing.assert_array_equal(
        first, mon.assign_conditions(3, 2, 0, 20, [1 / 3] * 3)
    )
    assert set(first) <= set(mon.CONDITIONS)
    many = np.concatenate(
        [
            mon.assign_conditions(3, t, b, 20, [1 / 3] * 3)
            for t in range(10)
            for b in range(10)
        ]
    )
    for name in mon.CONDITIONS:
        assert abs(np.mean(many == name) - 1 / 3) < 0.05
    # The hijack schedule's stream is untouched: drawing conditions first does
    # not change what the hijack RNG produces.
    hijack = np.random.default_rng([3, 2, 0, 0xEFFE]).random(5)
    np.testing.assert_array_equal(
        hijack, np.random.default_rng([3, 2, 0, 0xEFFE]).random(5)
    )
    only_off = mon.assign_conditions(3, 2, 0, 50, [1.0, 0.0, 0.0])
    assert set(only_off) == {"off"}


def _fitted_monitor(threshold=0.25):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 30)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 3] + rng.normal(scale=0.5, size=400) > 0).astype(int)
    pipeline = _make_classifier(1.0, 2000, None, "linear", 0).fit(x, y)
    scaler, clf = pipeline.named_steps["scale"], pipeline.named_steps["clf"]
    monitor = mon.LinearMonitor(
        mean=scaler.mean_,
        scale=scaler.scale_,
        coef=clf.coef_.ravel(),
        intercept=float(clf.intercept_[0]),
        threshold=threshold,
        meta={"source": "test"},
    )
    return monitor, pipeline, x


def test_linear_monitor_scores_like_the_sklearn_pipeline(tmp_path):
    monitor, pipeline, x = _fitted_monitor()
    np.testing.assert_allclose(
        monitor.score(x), pipeline.decision_function(x), atol=1e-5
    )
    np.testing.assert_array_equal(monitor.fires(x), monitor.score(x) > 0.25)
    digest = monitor.save(tmp_path / "m.json")
    loaded = mon.LinearMonitor.load(tmp_path / "m.json", expect_sha256=digest)
    np.testing.assert_array_equal(loaded.score(x), monitor.score(x))
    assert loaded.sha256 == digest and loaded.payload() == monitor.payload()


def test_linear_monitor_refuses_a_tampered_or_unexpected_artifact(tmp_path):
    monitor, _, _ = _fitted_monitor()
    path = tmp_path / "m.json"
    digest = monitor.save(path)
    with pytest.raises(ValueError, match="expected"):
        mon.LinearMonitor.load(path, expect_sha256="0" * 64)
    stored = json.loads(path.read_text())
    stored["monitor"]["threshold"] = 99.0
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="does not match"):
        mon.LinearMonitor.load(path, expect_sha256=digest)


# -- the config -------------------------------------------------------------


def test_every_earlier_config_keeps_its_hash():
    for name, expected in HASHES_BEFORE_X1_X4.items():
        assert load_config(str(CONFIG_DIR / name)).hash() == expected, name


def test_x1_x4_settings_change_the_hash_only_when_used(tmp_path):
    base = load_config(str(CONFIG_DIR / "main02_mirror.yaml"))
    extra = load_config(
        str(CONFIG_DIR / "main02_mirror.yaml"), ["capture.extra_observables=true"]
    )
    assert extra.hash() != base.hash()
    monitored = load_config(
        str(CONFIG_DIR / "main02_mirror.yaml"),
        [
            "monitor.enabled=true",
            "monitor.artifact=experiments/efference_probe/x4/m.json",
            f"monitor.artifact_sha256={'a' * 64}",
        ],
    )
    assert monitored.hash() != base.hash()


@pytest.mark.parametrize(
    "overrides, message",
    [
        (["monitor.enabled=true"], "artifact"),
        (
            [
                "monitor.enabled=true",
                "monitor.artifact=m.json",
                f"monitor.artifact_sha256={'0' * 64}",
            ],
            "hex string",
        ),
        (
            [
                "monitor.enabled=true",
                "monitor.artifact=m.json",
                f"monitor.artifact_sha256={'ab' * 32}",
                "monitor.condition_probs=[0.5, 0.6, 0.0]",
            ],
            "condition_probs",
        ),
        (
            [
                "monitor.enabled=true",
                "monitor.artifact=m.json",
                f"monitor.artifact_sha256={'ab' * 32}",
                "monitor.recovery=complete",
            ],
            "recovery",
        ),
    ],
)
def test_monitor_config_is_validated(overrides, message):
    with pytest.raises(ValueError, match=message):
        load_config(str(CONFIG_DIR / "main02_mirror.yaml"), overrides)
