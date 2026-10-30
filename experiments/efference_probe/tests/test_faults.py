"""X6 stage 0: actuation faults (faults.py) and their wiring into the harness."""

from __future__ import annotations

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
