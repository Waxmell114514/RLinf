"""The real collection loop (`EfferenceHarness._run_batch`) on a fake env.

No GPU, LIBERO or checkpoint: the model, the hidden-state capture and the env
are small fakes with the interfaces the harness uses.  This is the offline
check PREREG_X1_X4.md §4 asks for: the X1/X4 additions behave as specified,
and with their flags off the loop writes exactly what it wrote before.
"""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("omegaconf")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efference_probe import monitor as mon  # noqa: E402
from efference_probe.config import load_config  # noqa: E402
from efference_probe.harness import EfferenceHarness  # noqa: E402
from efference_probe.hijack import apply_mirror  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "main02_mirror.yaml"
N_ENVS, CHUNKS, DIM, N_CALLS = 4, 8, 7, 14
OBJECTS = {"bowl_1": 3, "plate_1": 3}


@dataclass
class FakeCaptured:
    hidden: np.ndarray
    entropy_mean: np.ndarray
    entropy_tokens: np.ndarray
    vision: None = None
    meta: dict = field(default_factory=dict)


class FakeCapture:
    def armed(self):
        return contextlib.nullcontext()

    def take(self):
        return FakeCaptured(
            hidden=np.zeros((N_ENVS, 1, 1, 4), np.float16),
            entropy_mean=np.zeros(N_ENVS, np.float32),
            entropy_tokens=np.zeros((N_ENVS, CHUNKS * DIM), np.float32),
        )


class FakeModel:
    """Deterministic chunks in the action bounds; gripper in the env's +/-1."""

    def __init__(self):
        self.calls = 0

    def predict_action_batch(self, env_obs, **_sampling):
        rng = np.random.default_rng(self.calls)
        self.calls += 1
        actions = rng.uniform(-0.9, 0.9, size=(N_ENVS, CHUNKS, DIM)).astype(np.float32)
        actions[..., 6] = np.sign(actions[..., 6]) + (actions[..., 6] == 0)
        result = {"prev_logprobs": torch.zeros(N_ENVS, CHUNKS * DIM)}
        return torch.from_numpy(actions), result


class FakeEnv:
    """Integrates OSC-like deltas; keeps a raw observation per env.

    ``eef_offset`` desynchronises the raw observation from ``states`` to check
    that the harness notices.
    """

    def __init__(self, eef_offset: float = 0.0, success_slot_call: tuple = (0, 9)):
        self.eef_offset = eef_offset
        self.success_slot, self.success_call = success_slot_call
        self.current_raw_obs = None

    def _obs(self):
        raw = []
        for i in range(N_ENVS):
            item = {
                "robot0_eef_pos": self.state[i, :3].astype(np.float64)
                + self.eef_offset,
                "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "robot0_gripper_qpos": self.state[i, 6:8].astype(np.float64),
                "robot0_joint_pos": self.joints[i].copy(),
            }
            start = 0
            for name, width in OBJECTS.items():
                item[f"{name}_pos"] = self.objects[i, start : start + width].copy()
                item[f"{name}_quat"] = np.array([0.0, 0.0, 0.0, 1.0])
                item[f"{name}_to_robot0_eef_pos"] = np.zeros(3)
                start += width
            raw.append(item)
        self.current_raw_obs = raw
        return {
            "states": torch.from_numpy(self.state.copy()),
            "main_images": torch.zeros(N_ENVS, 8, 8, 3, dtype=torch.uint8),
            "wrist_images": torch.zeros(N_ENVS, 8, 8, 3, dtype=torch.uint8),
        }

    def reset(self, env_idx, reset_state_ids):
        rng = np.random.default_rng(int(np.sum(reset_state_ids)))
        self.state = rng.normal(scale=0.1, size=(N_ENVS, 8)).astype(np.float32)
        self.state[:, 3:6] = 0.0
        self.joints = rng.normal(size=(N_ENVS, 7))
        self.objects = rng.normal(size=(N_ENVS, sum(OBJECTS.values())))
        self.n_chunks = 0
        return self._obs(), {}

    def chunk_step(self, executed):
        executed = np.asarray(executed, dtype=np.float32)
        self.state[:, :3] += 0.01 * executed[:, :, :3].sum(axis=1)
        self.joints += 0.01 * executed[:, :, :6].sum(axis=1).mean(axis=1, keepdims=True)
        self.objects[:, 0] += 0.001 * executed[:, :, 0].sum(axis=1)
        terminations = torch.zeros(N_ENVS, CHUNKS, dtype=torch.bool)
        if self.n_chunks == self.success_call:
            terminations[self.success_slot, -1] = True
        self.n_chunks += 1
        rewards = torch.zeros(N_ENVS, CHUNKS)
        truncations = torch.zeros(N_ENVS, CHUNKS, dtype=torch.bool)
        return [self._obs()], rewards, terminations, truncations, {}


def _harness(tmp_path, overrides=(), env=None, monitor=None):
    cfg = load_config(
        str(CONFIG),
        [
            f"out_root={tmp_path}",
            "run_id=fake",
            f"env.num_envs={N_ENVS}",
            f"env.max_calls_per_episode={N_CALLS}",
            "hijack.p_hijack=0.5",
            "hijack.probe_episode_fraction=1.0",
            "capture.verify_indexing=false",
            "frames.mode=none",
            *overrides,
        ],
    )
    harness = EfferenceHarness(cfg)
    harness.model, harness.capture = FakeModel(), FakeCapture()
    harness.env = env or FakeEnv()
    harness.monitor = monitor
    # The real mapping needs rlinf; the fake model already emits env-space chunks.
    harness._to_env_action_space = lambda commanded: np.asarray(commanded).copy()
    return harness


def _collect(harness):
    harness._run_batch(task_id=2, batch_ids=np.arange(N_ENVS), batch_index=0)
    harness.writer.finalize(harness._manifest_extra())
    calls = pd.read_parquet(harness.out_dir / "calls.parquet")
    manifest = pd.read_json(harness.out_dir / "manifest.json", typ="series")
    return calls, manifest


def _monitor(threshold):
    return mon.LinearMonitor(
        mean=np.zeros(31),
        scale=np.ones(31),
        coef=np.ones(31) * 1e-3,
        intercept=0.0,
        threshold=threshold,
    )


# -- flags off: the loop is unchanged ------------------------------------------

OLD_COLUMNS = {
    "run_id", "episode_id", "task_id", "reset_state_id", "env_slot", "call_idx",
    "is_probe_episode", "label", "transform", "donor_env_slot", "donor_episode_id",
    "skipped_reason", "a_cmd_model", "a_cmd_env", "a_exec", "states_before",
    "states_after", "logprob_sum", "logprobs", "entropy_mean", "entropy_tokens",
    "reward_delta", "success_substep", "terminated", "truncated",
    "first_success_call", "success_flag", "post_success",
}  # fmt: skip


def test_flags_off_write_the_old_columns_and_the_old_schedule(tmp_path):
    calls, manifest = _collect(_harness(tmp_path))
    assert set(calls.columns) == OLD_COLUMNS
    assert set(calls["label"]) == {"SELF", "HIJACK"}
    hijacked = calls[calls["label"] == "HIJACK"]
    assert len(hijacked) > 0 and (hijacked["call_idx"] >= 4).all()
    for _, row in hijacked.iterrows():
        cmd = np.asarray(row["a_cmd_env"]).reshape(CHUNKS, DIM)
        np.testing.assert_array_equal(
            np.asarray(row["a_exec"]).reshape(CHUNKS, DIM), apply_mirror(cmd)
        )
    assert "object_keys" not in manifest and "monitor" not in manifest
    # The success at call 9 stops slot 0's recording, as before.
    assert calls[calls["env_slot"] == 0]["call_idx"].max() == 9


# -- X1: extra observables ----------------------------------------------------


def test_extra_observables_are_logged_in_step_with_the_state(tmp_path):
    harness = _harness(tmp_path, ["capture.extra_observables=true"])
    calls, manifest = _collect(harness)
    assert manifest["object_keys"] == {
        "2": ["bowl_1_pos", "bowl_1_quat", "plate_1_pos", "plate_1_quat"]
    }
    joints = np.stack(calls["joint_pos_after"].map(np.asarray).to_list())
    objects = np.stack(calls["objects_after"].map(np.asarray).to_list())
    assert joints.shape == (len(calls), 7) and objects.shape == (len(calls), 14)
    # Each call's "before" is the previous call's "after".
    for _, episode in calls.groupby("episode_id"):
        episode = episode.sort_values("call_idx")
        for column in ("joint_pos", "objects"):
            before = np.stack(episode[f"{column}_before"].map(np.asarray).to_list())
            after = np.stack(episode[f"{column}_after"].map(np.asarray).to_list())
            np.testing.assert_array_equal(before[1:], after[:-1])


def test_extra_observables_refuse_a_raw_observation_out_of_step(tmp_path):
    harness = _harness(
        tmp_path, ["capture.extra_observables=true"], env=FakeEnv(eef_offset=1e-3)
    )
    with pytest.raises(RuntimeError, match="out of step"):
        harness._run_batch(task_id=2, batch_ids=np.arange(N_ENVS), batch_index=0)


# -- X4: monitor, oracle and the undo ------------------------------------------


def _monitor_overrides(probs):
    return [
        "monitor.enabled=true",
        "monitor.artifact=unused.json",
        f"monitor.artifact_sha256={'ab' * 32}",
        f"monitor.condition_probs={probs}",
    ]


def _check_recoveries(calls):
    """Every fire is followed by an undo of the chunk it fired on."""
    for _, episode in calls.groupby("episode_id"):
        episode = episode.sort_values("call_idx").reset_index(drop=True)
        for i, row in episode.iterrows():
            if i + 1 >= len(episode):
                break
            following = episode.iloc[i + 1]
            assert following["is_recovery"] == bool(row["monitor_fired"])
            if row["monitor_fired"]:
                assert following["label"] == "RECOVERY"
                executed = np.asarray(row["a_exec"]).reshape(CHUNKS, DIM)
                np.testing.assert_array_equal(
                    np.asarray(following["a_exec"]).reshape(CHUNKS, DIM),
                    mon.undo_chunk(executed),
                )
    recovery = calls[calls["is_recovery"]]
    assert recovery["monitor_score"].isna().all()
    assert not recovery["monitor_fired"].any()


def test_oracle_undoes_exactly_the_hijacked_chunks(tmp_path):
    harness = _harness(
        tmp_path, _monitor_overrides("[0.0, 0.0, 1.0]"), monitor=_monitor(0.0)
    )
    calls, manifest = _collect(harness)
    assert (calls["monitor_condition"] == "oracle").all()
    np.testing.assert_array_equal(calls["monitor_fired"], calls["label"] == "HIJACK")
    assert calls["is_recovery"].sum() > 0
    _check_recoveries(calls)
    # A recovery call is never hijacked, and scoring starts at call 4.
    assert set(calls.loc[calls["is_recovery"], "label"]) == {"RECOVERY"}
    early = calls[calls["call_idx"] < 4]
    assert early["monitor_score"].isna().all()
    episodes = pd.DataFrame(manifest["episodes"])
    assert set(episodes["monitor_condition"]) == {"oracle"}
    assert episodes["n_recoveries"].sum() == calls["is_recovery"].sum()


@pytest.mark.parametrize("threshold, expect_fires", [(-np.inf, True), (np.inf, False)])
def test_monitor_fires_by_its_threshold(tmp_path, threshold, expect_fires):
    harness = _harness(
        tmp_path, _monitor_overrides("[0.0, 1.0, 0.0]"), monitor=_monitor(threshold)
    )
    calls, _ = _collect(harness)
    scored = calls[calls["monitor_score"].notna()]
    assert len(scored) > 0
    assert scored["monitor_fired"].all() == expect_fires
    assert scored["monitor_fired"].any() == expect_fires
    _check_recoveries(calls)


def test_logged_scores_recompute_offline_from_the_logged_rows(tmp_path):
    monitor = _monitor(0.0)
    harness = _harness(tmp_path, _monitor_overrides("[0.0, 1.0, 0.0]"), monitor=monitor)
    calls, _ = _collect(harness)
    scored = calls[calls["monitor_score"].notna()]
    commanded = np.stack(scored["a_cmd_env"].map(np.asarray).to_list()).reshape(
        -1, CHUNKS, DIM
    )
    before = np.stack(scored["states_before"].map(np.asarray).to_list())
    after = np.stack(scored["states_after"].map(np.asarray).to_list())
    offline = monitor.score(mon.mismatch_features(commanded, before, after))
    np.testing.assert_allclose(scored["monitor_score"], offline, atol=1e-6)


def test_off_condition_scores_but_never_acts(tmp_path):
    harness = _harness(
        tmp_path, _monitor_overrides("[1.0, 0.0, 0.0]"), monitor=_monitor(-np.inf)
    )
    calls, _ = _collect(harness)
    assert calls["monitor_score"].notna().any()
    assert not calls["monitor_fired"].any() and not calls["is_recovery"].any()
    assert set(calls["label"]) == {"SELF", "HIJACK"}
