"""Offline tests for the GRU mismatch experiment (CPU, a few seconds).

Run with ``pytest experiments/gru_mismatch/tests -q``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

torch = pytest.importorskip("torch")

from gru_mismatch.analyze import analyze_arm, build_samples  # noqa: E402
from gru_mismatch.mismatch import (  # noqa: E402
    HIJACK,
    SELF,
    ArmData,
    MismatchConfig,
    collect,
    execute,
)
from gru_mismatch.port import GRUAgent, ProtocolConfig, train  # noqa: E402

TINY = {
    "train_steps": 200,
    "phase2_steps": 300,
    "burst_mean": 30,
    "burst_min": 15,
    "burst_max": 60,
    "trailing_steps": 5,
}


@pytest.fixture(scope="module")
def trained_command():
    return train(ProtocolConfig(interface="command", seed=3, **TINY))


@pytest.fixture(scope="module")
def arms(trained_command):
    config = MismatchConfig(steps=1500)
    return {
        t: collect(trained_command, t, config) for t in ("swap", "mirror", "freeze")
    }


def test_execute_transforms():
    assert execute("swap", 1.5, -0.3) == -0.3
    assert execute("mirror", 1.5, -0.3) == -1.5
    assert execute("freeze", 1.5, -0.3) == 0.0
    with pytest.raises(ValueError):
        execute("teleport", 1.0, 0.0)


def test_interface_widths():
    assert GRUAgent("none").gru.input_size == 4
    for interface in ("trace_abs", "trace_signed", "command"):
        assert GRUAgent(interface).gru.input_size == 5
    assert GRUAgent("both").gru.input_size == 6
    with pytest.raises(ValueError):
        ProtocolConfig(interface="telepathy")


def test_both_interface_feeds_abs_trace_and_signed_command():
    agent = GRUAgent("both")
    agent.observe_command(-1.0)
    x = agent._gru_input(torch.zeros(4))
    assert x.shape == (1, 6)
    assert x[0, 4].item() == pytest.approx(0.05)  # |a| trace, as in exp4
    assert x[0, 5].item() == pytest.approx(-1.0)  # signed instantaneous command
    assert agent.efference_value == pytest.approx(-1.0)


def test_trace_sign_handling():
    absolute, signed = GRUAgent("trace_abs"), GRUAgent("trace_signed")
    for agent in (absolute, signed):
        agent.observe_command(-1.0)
    assert absolute.trace == pytest.approx(0.05)
    assert signed.trace == pytest.approx(-0.05)


def test_arms_are_paired(arms):
    """Same world in every arm: only hijacked executed actions differ."""
    base = arms["swap"].steps
    for transform in ("mirror", "freeze"):
        other = arms[transform].steps
        for column in ("label", "command", "segment", "since_onset", "active"):
            assert (base[column].to_numpy() == other[column].to_numpy()).all(), column
        clean = other["label"] == SELF
        assert np.allclose(other.loc[clean, "executed"], other.loc[clean, "command"])
    hijacked = base["label"] == HIJACK
    assert hijacked.sum() > 20
    mirror, freeze = arms["mirror"].steps, arms["freeze"].steps
    assert np.allclose(
        mirror.loc[hijacked, "executed"], -mirror.loc[hijacked, "command"]
    )
    assert np.allclose(freeze.loc[hijacked, "executed"], 0.0)


def test_schedule_rules(arms):
    steps = arms["swap"].steps
    config = MismatchConfig()
    hijacked = steps[steps["label"] == HIJACK]
    assert hijacked["active"].all()
    assert (hijacked["since_onset"] >= config.warmup).all()
    gaps = np.diff(hijacked["index"].to_numpy())
    assert (gaps > config.min_clean_gap).all()
    quiet = steps[~steps["active"]]
    assert np.allclose(quiet["command"], 0.0)
    assert (quiet["label"] == SELF).all()


def test_efference_carries_the_command_not_the_outcome(arms):
    """The mismatch condition: tau gets a, the world gets gamma * a'."""
    steps = arms["swap"].steps
    hijacked = steps[steps["label"] == HIJACK]
    assert np.allclose(hijacked["efference"], hijacked["command"], atol=1e-6)
    assert not np.allclose(hijacked["executed"], hijacked["command"])


def test_samples_point_at_the_manipulated_step(arms):
    arm = arms["freeze"]
    samples = build_samples(arm, warmup=4, min_clean_gap=2)
    assert not samples.empty
    labels = arm.steps.set_index("index")["label"]
    positive = samples[samples["y"] == 1]
    negative = samples[samples["y"] == 0]
    assert (labels.loc[positive["index"]] == HIJACK).all()
    assert (labels.loc[negative["index"]] == SELF).all()
    # Pairs never cross segments: the CV group is the segment.
    paired = samples[samples["pair_id"] >= 0]
    assert (paired.groupby("pair_id")["episode_id"].nunique() == 1).all()


def _planted_arm(n_segments: int = 40, length: int = 60, seed: int = 0) -> ArmData:
    """An arm whose hidden state carries the label at the hijacked step only."""
    rng = np.random.default_rng(seed)
    rows = []
    index = 0
    for segment in range(1, n_segments + 1):
        last = -10
        for since in range(length):
            hijacked = since >= 4 and since - last > 2 and rng.random() < 0.25
            if hijacked:
                last = since
            rows.append(
                {
                    "index": index,
                    "segment": segment,
                    "since_onset": since,
                    "active": True,
                    "cls": 0,
                    "command": rng.uniform(-2, 2),
                    "executed": 0.0,
                    "label": HIJACK if hijacked else SELF,
                    "transform": "freeze" if hijacked else "",
                    "trace": 0.0,
                    "efference": 0.0,
                    **{f"obs{k}": 0.0 for k in range(4)},
                    **{f"res{k}": rng.normal() for k in range(4)},
                }
            )
            index += 1
    steps = pd.DataFrame(rows)
    hidden = rng.normal(size=(len(steps), 16)).astype(np.float32)
    # Signal ONLY at the manipulated step: an off-by-one between samples and
    # states would read noise and land at chance.
    hidden[:, 0] += 3.0 * (steps["label"] == HIJACK).to_numpy()
    return ArmData(transform="freeze", steps=steps, h_multi=hidden, h_gru=hidden.copy())


def test_planted_signal_is_recovered_and_floor_is_chance():
    table, summary = analyze_arm(_planted_arm(), warmup=4, min_clean_gap=2, gamma=2.0)
    score = dict(zip(table["name"], table["balanced_acc_mean"]))
    assert score["P1"] > 0.85
    assert score["P0"] < 0.6
    assert score["C_cmd"] < 0.6
    assert summary["margin"] > 0.25
