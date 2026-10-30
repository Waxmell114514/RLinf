"""X6 stage 0: actuation faults (faults.py) and their wiring into the harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efference_probe import faults  # noqa: E402
from efference_probe.config import load_config  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "main02_mirror.yaml"


def test_each_family_changes_only_its_dims_and_stays_in_bounds():
    rng = np.random.default_rng(0)
    chunk = rng.uniform(-0.9, 0.9, size=(3, 8, 7)).astype(np.float32)
    np.testing.assert_array_equal(faults.parse("clean").apply(chunk), chunk)
    inverted = faults.parse("invert_x").apply(chunk)
    np.testing.assert_array_equal(inverted[..., 0], -chunk[..., 0])
    np.testing.assert_array_equal(inverted[..., 1:], chunk[..., 1:])
    biased = faults.parse("bias_x:0.3").apply(chunk)
    np.testing.assert_allclose(biased[..., 0], np.clip(chunk[..., 0] + 0.3, -1, 1))
    np.testing.assert_array_equal(biased[..., 1:], chunk[..., 1:])
    gained = faults.parse("gain:0.6").apply(chunk)
    np.testing.assert_allclose(gained[..., :3], chunk[..., :3] * 0.6)
    np.testing.assert_array_equal(gained[..., 3:], chunk[..., 3:])
    for out in (inverted, biased, gained):
        assert np.abs(out[..., :6]).max() <= 1.0
    assert chunk.dtype == np.float32 and not np.shares_memory(biased, chunk)


@pytest.mark.parametrize(
    "bad", ["bias_x", "bias_x:1.2", "gain:0", "invert_x:1", "drift"]
)
def test_bad_specs_are_refused(bad):
    with pytest.raises(ValueError):
        faults.parse(bad)


def test_fault_config_leaves_old_hashes_and_refuses_hijack():
    base = load_config(str(CONFIG))
    assert base.hash() == load_config(str(CONFIG), ["fault.enabled=false"]).hash()
    with pytest.raises(ValueError, match="hijack and monitor"):
        load_config(str(CONFIG), ["fault.enabled=true", "fault.sweep=[clean]"])
    cfg = load_config(
        str(CONFIG),
        ["hijack.enabled=false", "fault.enabled=true", "fault.sweep=[clean, invert_x]"],
    )
    assert cfg.fault.sweep == ["clean", "invert_x"] and cfg.hash() != base.hash()


def test_sweep_runs_every_fault_on_the_same_states(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("omegaconf")
    from efference_probe.tests import test_x1_x4_harness as fake

    harness = fake._harness(
        tmp_path,
        [
            "hijack.enabled=false",
            "fault.enabled=true",
            "fault.sweep=[clean, bias_x:0.3, invert_x]",
            "env.task_ids=[2]",
            f"env.init_states_per_task={fake.N_ENVS}",
        ],
    )
    harness.setup = lambda: None  # the fakes are already in place
    harness.teardown = lambda: None
    harness.reset_state_ids_for_task = lambda task: np.arange(fake.N_ENVS)
    harness.run()
    calls = pd.read_parquet(harness.out_dir / "calls.parquet")
    manifest = pd.read_json(harness.out_dir / "manifest.json", typ="series")
    episodes = pd.DataFrame(manifest["episodes"])
    assert list(episodes.groupby("fault").size()) == [fake.N_ENVS] * 3
    for spec, group in calls.groupby("fault"):
        fault = faults.parse(spec)
        for _, row in group.iterrows():
            cmd = np.asarray(row["a_cmd_env"]).reshape(fake.CHUNKS, fake.DIM)
            np.testing.assert_allclose(
                np.asarray(row["a_exec"]).reshape(fake.CHUNKS, fake.DIM),
                fault.apply(cmd),
            )
    # Same reset states under every fault.
    states = episodes.groupby("fault")["episode_id"].count()
    assert (
        set(calls.groupby("fault")["reset_state_id"].apply(frozenset))
        == {frozenset(range(fake.N_ENVS))}
        and len(states) == 3
    )
    assert set(calls["label"]) == {"SELF"}


# -- X6 fault benchmark across architectures ---------------------------------

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_benchmark_configs_build_their_fault_and_model_sections():
    from efference_probe.harness import build_model_config

    oft = load_config(str(CONFIGS / "x6_fault_oft.yaml"))
    pi0 = load_config(str(CONFIGS / "x6_fault_pi0.yaml"))
    for cfg in (oft, pi0):
        assert cfg.fault.enabled and len(cfg.fault.sweep) == 6
        assert not cfg.capture.enabled and not cfg.hijack.enabled
        # Every architecture gets the env yaml's 320 sim steps.
        assert cfg.env.max_calls_per_episode * cfg.model.num_action_chunks == 320
    model = build_model_config(pi0)
    assert model.model_type == "openpi" and model.num_action_chunks == 5
    assert model.openpi.action_chunk == 5
    assert "implement_version" not in model  # no OFT fields leak in
    assert build_model_config(oft).implement_version == "rlinf"


def test_capture_off_rules():
    with pytest.raises(ValueError, match="verify_indexing"):
        load_config(str(CONFIG), ["capture.enabled=false"])
    with pytest.raises(ValueError, match="openvla_oft only"):
        load_config(str(CONFIG), ["model.model_type=openpi"])


def test_capture_off_loop_logs_behaviour_without_logprobs(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("omegaconf")
    from efference_probe.tests import test_x1_x4_harness as fake

    class ModeOnlyModel(fake.FakeModel):
        """Another architecture: takes `mode`, returns no logprobs."""

        seen: list = []

        def predict_action_batch(self, env_obs, **kwargs):
            ModeOnlyModel.seen.append(kwargs)
            actions, _ = super().predict_action_batch(env_obs)
            return actions, {}

    harness = fake._harness(
        tmp_path,
        [
            "hijack.enabled=false",
            "capture.enabled=false",
            "capture.verify_indexing=false",
            "model.predict_mode=eval",
            "fault.enabled=true",
            "fault.sweep=[bias_x:0.3]",
        ],
    )
    harness.model, harness.capture = ModeOnlyModel(), None
    harness._fault = faults.parse("bias_x:0.3")
    calls, _ = fake._collect(harness)
    assert ModeOnlyModel.seen and all(k == {"mode": "eval"} for k in ModeOnlyModel.seen)
    assert calls["logprob_sum"].isna().all() and calls["entropy_mean"].isna().all()
    assert not (harness.out_dir / "hidden").exists() or not any(
        (harness.out_dir / "hidden").iterdir()
    )
    assert (calls["fault"] == "bias_x:0.3").all()


def test_fault_readout_pairs_states_and_counts_failures_at_the_budget(tmp_path):
    from efference_probe import x6_fault_report as report

    run = tmp_path / "run"
    run.mkdir()
    episodes, rows = [], []
    for i, (fault, success) in enumerate(
        [("clean", True), ("clean", True), ("bias_x:0.3", False), ("bias_x:0.3", True)]
    ):
        state = i % 2
        episodes.append(
            {
                "episode_id": i,
                "task_id": 2,
                "fault": fault,
                "success": success,
                "first_success_call": 4 if success else -1,
            }
        )
        for call in range(3):
            rows.append(
                {
                    "episode_id": i,
                    "call_idx": call,
                    "reset_state_id": state,
                    "task_id": 2,
                    "a_cmd_env": np.full(5 * 7, 0.1 * call, np.float32),
                }
            )
    (run / "manifest.json").write_text(json.dumps({"episodes": episodes}))
    pd.DataFrame(rows).to_parquet(run / "calls.parquet")
    out = report.readout(run)
    assert out["chunk"] == 5
    clean, biased = out["faults"]["clean"], out["faults"]["bias_x:0.3"]
    assert clean["rmst_steps"]["point"] == 25  # (4 + 1) x 5 steps
    assert biased["rmst_steps"]["point"] == (320 + 25) / 2
    assert biased["drop"]["point"] == 0.5 and biased["drop"]["n_states"] == 2
    np.testing.assert_allclose(clean["reaction_x"], [0.0, 0.1, 0.2], atol=1e-6)
