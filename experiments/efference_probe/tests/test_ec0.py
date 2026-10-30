"""EC0's comparator on synthetic dynamics (logbook, "EC0 PRE-REGISTRATION")."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efference_probe import ec0_precheck as ec0  # noqa: E402


def _world(n_episodes=60, calls=20, hijack_rate=0.2, seed=0, hidden_noise=0.05):
    """Linear dynamics: s(c+1) = s(c) + B a_exec; a hijack executes another chunk."""
    rng = np.random.default_rng(seed)
    gain = rng.normal(size=(56, 8)) * 0.05
    rows = []
    for episode in range(n_episodes):
        state = rng.normal(size=8)
        for call in range(calls):
            commanded = rng.normal(size=(8, 7))
            hijack = call >= 4 and rng.random() < hijack_rate
            executed = rng.normal(size=(8, 7)) if hijack else commanded
            after = state + executed.reshape(-1) @ gain
            rows.append((episode, episode % 3, state, after, commanded, hijack))
            state = after
    groups, tasks, before, after, commands, hijack = map(np.array, zip(*rows))
    # 3D rotations stay small, so compose_delta acts like a difference.
    before, after = before * 0.1, after * 0.1
    mix = rng.normal(size=(8, 32))
    h_before = before @ mix + rng.normal(size=(len(groups), 32)) * hidden_noise
    h_after = after @ mix + rng.normal(size=(len(groups), 32)) * hidden_noise
    return groups, tasks, before, after, commands, hijack, h_before, h_after


def test_comparator_detects_overrides_from_states_and_from_decodes():
    groups, tasks, before, after, commands, hijack, h_b, h_a = _world()
    is_self = ~hijack
    true = ec0.oof_scores(before, after, commands, tasks, is_self, groups)
    point = ec0.operating_point(true, is_self, hijack)
    assert abs(point["fpr"] - ec0.FPR) < 0.005 and point["tpr"] > 0.9
    feat = ec0.oof_scores(before, after, commands, tasks, is_self, groups, h_b, h_a)
    assert np.isfinite(feat).all()
    assert ec0.operating_point(feat, is_self, hijack)["auroc"] > 0.8


def test_no_hijack_row_enters_any_fit(monkeypatch):
    groups, tasks, before, after, commands, hijack, h_b, h_a = _world(n_episodes=30)
    fitted = []
    real = ec0.x0._Decoder

    class Spy(real):
        def __init__(self, x, y, g):
            fitted.append(len(x))
            super().__init__(x, y, g)

    monkeypatch.setattr(ec0.x0, "_Decoder", Spy)
    # Poison every HIJACK row: any fit that saw one would return NaN.
    before_p, h_b_p = before.copy(), h_b.copy()
    before_p[hijack], h_b_p[hijack] = np.nan, np.nan
    scores = ec0.oof_scores(
        before_p, after, commands, tasks, ~hijack, groups, h_b_p, h_a
    )
    assert fitted and np.isfinite(scores[~hijack]).all()


def test_reading_follows_the_rule():
    def result(tpr):
        return {"V_feat": {"tpr": tpr}}

    assert ec0.reading({r: result(0.6) for r in ec0.GATED}) == "go"
    mixed = {r: result(0.6 if i < 2 else 0.3) for i, r in enumerate(ec0.GATED)}
    assert ec0.reading(mixed) == "mixed"
    assert ec0.reading({r: result(0.1) for r in ec0.GATED}) == "no-go"
    assert ec0.reading({ec0.GATED[0]: result(0.9)}).startswith("unresolved")


@pytest.mark.parametrize("fpr", [0.02])
def test_operating_point_holds_the_false_positive_rate(fpr):
    rng = np.random.default_rng(1)
    scores = rng.random(1000)
    is_self = np.arange(1000) < 900
    point = ec0.operating_point(scores, is_self, ~is_self)
    assert abs(point["fpr"] - fpr) < 0.005
