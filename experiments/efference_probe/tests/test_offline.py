"""Offline tests: everything that does not need a GPU, LIBERO, or a checkpoint.

Run with ``pytest experiments/efference_probe/tests/test_offline.py``.  These
cover the hijack schedule, the probe-sample construction, the storage
round-trip, and the sequence arithmetic behind the hidden-state pools -- the
parts where a bug would silently poison a run that costs GPU hours.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efference_probe.analysis import to_markdown  # noqa: E402
from efference_probe.config import RunConfig, load_config  # noqa: E402
from efference_probe.datasets import (  # noqa: E402
    HiddenStore,
    RunData,
    build_features,
    build_probe_samples,
    load_run,
    step_index_report,
)
from efference_probe.hijack import (  # noqa: E402
    GRIPPER_INDEX,
    HijackScheduler,
    apply_freeze,
    apply_mirror,
    build_executed_chunk,
)
from efference_probe.probes import run_probe  # noqa: E402
from efference_probe.storage import DiskBudget, RunWriter, flatten  # noqa: E402
from efference_probe.tests.synthetic import make_synthetic_run  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


# --------------------------------------------------------------------------
# Hijack transforms and schedule
# --------------------------------------------------------------------------


def test_mirror_negates_only_xy():
    chunk = np.arange(8 * 7, dtype=np.float32).reshape(8, 7)
    mirrored = apply_mirror(chunk)
    assert np.allclose(mirrored[:, 0:2], -chunk[:, 0:2])
    assert np.allclose(mirrored[:, 2:], chunk[:, 2:])
    # The input must survive untouched: a_cmd is logged from it.
    assert np.allclose(chunk, np.arange(8 * 7).reshape(8, 7))


def test_freeze_zeros_deltas_and_holds_gripper():
    chunk = np.random.default_rng(0).normal(size=(8, 7)).astype(np.float32)
    frozen = apply_freeze(chunk)
    assert np.allclose(frozen[:, :6], 0.0)
    assert np.allclose(frozen[:, GRIPPER_INDEX], chunk[:, GRIPPER_INDEX])


def test_scheduler_respects_warmup_and_gap():
    rng = np.random.default_rng(3)
    scheduler = HijackScheduler(
        num_envs=6,
        p_hijack=1.0,
        warmup_calls=4,
        min_clean_gap=2,
        transform="freeze",
        rng=rng,
    )
    active = np.ones(6, dtype=bool)
    hijacked_calls = {env: [] for env in range(6)}
    for call_idx in range(20):
        for decision in scheduler.decide(call_idx, active):
            if decision.hijacked:
                hijacked_calls[decision.env_index].append(call_idx)

    for calls in hijacked_calls.values():
        assert calls, "p_hijack=1.0 must produce hijacks"
        assert min(calls) >= 4, "no hijack may land during warm-up"
        gaps = np.diff(calls)
        assert (gaps >= 3).all(), "two clean calls must follow every hijack"


def test_scheduler_rate_is_approximately_p():
    """With the gap constraint the realised rate is below p; it must still track it."""
    rng = np.random.default_rng(0)
    scheduler = HijackScheduler(
        num_envs=200,
        p_hijack=0.25,
        warmup_calls=4,
        min_clean_gap=2,
        transform="freeze",
        rng=rng,
    )
    active = np.ones(200, dtype=bool)
    eligible = hijacked = 0
    for call_idx in range(60):
        eligible += int(scheduler.eligible(call_idx, active).sum())
        hijacked += sum(d.hijacked for d in scheduler.decide(call_idx, active))
    assert 0.2 < hijacked / eligible < 0.3


def test_scheduler_skips_swap_without_donor():
    scheduler = HijackScheduler(
        num_envs=2,
        p_hijack=1.0,
        warmup_calls=0,
        min_clean_gap=1,
        transform="swap",
        rng=np.random.default_rng(0),
    )
    active = np.array([True, False])
    decisions = scheduler.decide(0, active)
    assert not decisions[0].hijacked
    assert decisions[0].skipped_reason == "no_swap_donor"


def test_clean_episodes_can_still_donate():
    """An episode excluded from hijacking is still a valid swap donor."""
    scheduler = HijackScheduler(
        num_envs=2,
        p_hijack=1.0,
        warmup_calls=0,
        min_clean_gap=1,
        transform="swap",
        rng=np.random.default_rng(0),
    )
    # Env 0 may be hijacked; env 1 is a clean control episode but still live.
    decisions = scheduler.decide(
        0, active=np.array([True, False]), donor_active=np.array([True, True])
    )
    assert decisions[0].hijacked
    assert decisions[0].donor_env_index == 1
    assert not decisions[1].hijacked


def test_swap_reads_pre_hijack_commands():
    """A chain of swaps must never propagate an already-hijacked chunk."""
    commanded = np.stack(
        [np.full((4, 7), value, dtype=np.float32) for value in (1.0, 2.0, 3.0)]
    )
    from efference_probe.hijack import HijackDecision

    decisions = [
        HijackDecision(0, True, "swap", donor_env_index=1),
        HijackDecision(1, True, "swap", donor_env_index=2),
        HijackDecision(2, False),
    ]
    executed = build_executed_chunk(commanded, decisions)
    assert np.allclose(executed[0], 2.0), "env 0 must get env 1's *original* chunk"
    assert np.allclose(executed[1], 3.0)
    assert np.allclose(executed[2], 3.0)
    assert np.allclose(commanded[0], 1.0), "commanded must not be mutated"


# --------------------------------------------------------------------------
# Sequence arithmetic behind the hidden-state pools
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_patches,n_prompt,n_act", [(256, 128, 56), (512, 64, 56)])
def test_readout_window_matches_model_slice(n_patches, n_prompt, n_act):
    """The two slices `predict_action_batch` uses are the same window.

    It reads logits at ``[n_patches + n_prompt_tokens : + n_act]`` and hidden
    states at ``[-n_act-1 : -1]``.  Those agree only because the sequence is
    ``BOS + patches + (n_prompt - 1) + n_act``.  If this identity ever breaks,
    the pools in `capture.py` are pointing at the wrong tokens.
    """
    n_prompt_tokens = n_prompt - 1
    seq_len = n_patches + n_prompt + n_act
    logits_start = n_patches + n_prompt_tokens
    hidden_start = seq_len - n_act - 1
    assert logits_start == hidden_start
    assert logits_start + n_act == seq_len - 1


# --------------------------------------------------------------------------
# Probe-sample construction
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    path = tmp_path_factory.mktemp("synthetic")
    make_synthetic_run(path, hidden_signal=1.2, seed=7)
    return load_run(path)


def test_samples_are_paired_and_well_formed(synthetic_run):
    samples = build_probe_samples(
        synthetic_run.calls,
        min_clean_gap=2,
        warmup_calls=4,
        rng=np.random.default_rng(0),
    )
    assert not samples.empty
    assert (samples["y"] == 1).sum() == (samples["y"] == 0).sum()
    assert (samples["cur_call"] == samples["prev_call"] + 1).all()
    # Every pair_id must hold exactly one positive and one negative.
    counts = samples.groupby("pair_id")["y"].agg(["count", "sum"])
    assert (counts["count"] == 2).all()
    assert (counts["sum"] == 1).all()
    # No call may serve as a negative twice.
    negatives = samples[samples["y"] == 0]
    assert not negatives.duplicated(subset=["episode_id", "cur_call"]).any()


def test_positives_follow_hijacks_and_negatives_do_not(synthetic_run):
    calls = synthetic_run.calls
    labels = {
        (int(e), int(c)): label
        for e, c, label in zip(calls["episode_id"], calls["call_idx"], calls["label"])
    }
    samples = build_probe_samples(
        synthetic_run.calls,
        min_clean_gap=2,
        warmup_calls=4,
        rng=np.random.default_rng(0),
    )
    for _, sample in samples.iterrows():
        key = (int(sample["episode_id"]), int(sample["prev_call"]))
        expected = "HIJACK" if sample["y"] == 1 else "SELF"
        assert labels[key] == expected
        # The call being probed is always self-caused, for both classes.
        assert labels[(int(sample["episode_id"]), int(sample["cur_call"]))] == "SELF"


def test_negatives_are_phase_matched(synthetic_run):
    """The classes' call-index distributions must overlap (SPEC 3.2)."""
    samples = build_probe_samples(
        synthetic_run.calls,
        min_clean_gap=2,
        warmup_calls=4,
        rng=np.random.default_rng(0),
    )
    report = step_index_report(samples)
    assert abs(report.loc[1, "mean"] - report.loc[0, "mean"]) < 2.0
    assert samples["phase_distance"].mean() < 6.0


def test_negatives_never_precede_the_earliest_possible_positive(synthetic_run):
    warmup = 4
    samples = build_probe_samples(
        synthetic_run.calls,
        min_clean_gap=2,
        warmup_calls=warmup,
        rng=np.random.default_rng(0),
    )
    assert samples["cur_call"].min() >= warmup + 1


def test_post_success_calls_are_excluded(synthetic_run):
    samples = build_probe_samples(
        synthetic_run.calls,
        min_clean_gap=2,
        warmup_calls=4,
        rng=np.random.default_rng(0),
    )
    calls = synthetic_run.calls.set_index(["episode_id", "call_idx"])
    for _, sample in samples.iterrows():
        row = calls.loc[(int(sample["episode_id"]), int(sample["cur_call"]))]
        assert not bool(row["post_success"])


# --------------------------------------------------------------------------
# Probe behaviour against known ground truth
# --------------------------------------------------------------------------


def _probe(run, samples, blocks, layer=None, pool=None, **kwargs):
    features, _ = build_features(
        run,
        samples,
        blocks,
        layer=layer,
        pool=pool,
        store=HiddenStore(run.run_dir / "hidden"),
    )
    return run_probe(
        features,
        samples["y"].to_numpy(),
        samples["episode_id"].to_numpy(),
        name="test",
        blocks=blocks,
        layer=layer,
        pool=pool,
        select_c=False,
        **kwargs,
    )


def test_shuffled_labels_sit_at_chance(synthetic_run):
    samples = build_probe_samples(
        synthetic_run.calls, warmup_calls=4, rng=np.random.default_rng(0)
    )
    result = _probe(
        synthetic_run,
        samples,
        ["h_cur"],
        layer=8,
        pool="act_mean",
        shuffle_labels=True,
        seed=0,
    )
    assert 0.35 < result.balanced_acc_mean < 0.65


def test_commanded_action_alone_is_not_predictive(synthetic_run):
    """The command is produced before the hijack, so it cannot leak the label."""
    samples = build_probe_samples(
        synthetic_run.calls, warmup_calls=4, rng=np.random.default_rng(0)
    )
    result = _probe(synthetic_run, samples, ["a_cmd"], seed=0)
    assert result.balanced_acc_mean < 0.65


def test_mismatch_oracle_recovers_the_label(synthetic_run):
    """The synthetic state really is driven by the executed chunk, so the
    comparison features must nearly separate the classes."""
    samples = build_probe_samples(
        synthetic_run.calls, warmup_calls=4, rng=np.random.default_rng(0)
    )
    result = _probe(synthetic_run, samples, ["mismatch"], seed=0)
    assert result.balanced_acc_mean > 0.85


def test_concatenation_oracle_fails_for_mean_preserving_transforms(tmp_path):
    """Why P2r exists, stated precisely.

    A linear model on ``a_cmd + dstates`` cannot form the agreement between
    them.  For a mean-*preserving* transform (swap: a donor chunk has the same
    action statistics) that leaves nothing else to separate on, so P2 sits at
    chance while the comparison features recover the label.
    """
    path = make_synthetic_run(
        tmp_path / "swap", transform="swap", command_mean=0.0, seed=11
    )
    run = load_run(path)
    samples = build_probe_samples(
        run.calls, warmup_calls=4, rng=np.random.default_rng(0)
    )
    concat = _probe(run, samples, ["a_cmd", "dstates"], seed=0)
    mismatch = _probe(run, samples, ["mismatch"], seed=0)
    assert concat.balanced_acc_mean < 0.65
    assert mismatch.balanced_acc_mean > 0.85


def test_concatenation_oracle_succeeds_for_mean_shifting_transforms(tmp_path):
    """The counterpart, and the reason P2 is not a ceiling either way.

    With directed motion, ``freeze`` shifts the mean of the state delta, so a
    linear probe separates the classes on the marginal alone -- without ever
    comparing command to outcome.  ``C_dstates`` rising with it is the tell.
    SPEC 5's ``P2 >= 0.9`` gate is therefore transform-dependent, and P2r is
    the gate to use.
    """
    path = make_synthetic_run(
        tmp_path / "freeze", transform="freeze", command_mean=0.25, seed=11
    )
    run = load_run(path)
    samples = build_probe_samples(
        run.calls, warmup_calls=4, rng=np.random.default_rng(0)
    )
    concat = _probe(run, samples, ["a_cmd", "dstates"], seed=0)
    dstates_only = _probe(run, samples, ["dstates"], seed=0)
    assert concat.balanced_acc_mean > 0.9
    assert dstates_only.balanced_acc_mean > 0.9


def test_planted_hidden_signal_is_recovered(synthetic_run):
    samples = build_probe_samples(
        synthetic_run.calls, warmup_calls=4, rng=np.random.default_rng(0)
    )
    result = _probe(synthetic_run, samples, ["h_cur"], layer=8, pool="act_mean", seed=0)
    assert result.balanced_acc_mean > 0.6


def test_undo_alignment_is_not_fooled_by_command_autocorrelation(tmp_path):
    """The metric must measure correction, not command persistence.

    Scoring the raw next command against the negated error reduces to
    ``+(A_prev . A_cur)``: for `freeze` the residual is exactly
    ``-gain * A_prev``.  So a policy that never looks at the state still scores
    a large positive "undo alignment" purely from temporal autocorrelation in
    its own commands -- and the SELF control cannot catch it, because for SELF
    samples the residual is noise and the control sits at zero regardless.
    """
    from efference_probe.analysis import undo_alignment

    path = make_synthetic_run(
        tmp_path / "ar1",
        transform="freeze",
        command_autocorr=0.9,
        seed=5,
        episodes_per_task=10,
        n_tasks=3,
        n_calls=24,
        p_hijack=0.3,
    )
    run = load_run(path)
    samples = build_probe_samples(
        run.calls, warmup_calls=4, rng=np.random.default_rng(0)
    )
    report = undo_alignment(run, samples)

    # The generator's policy is open-loop, so the true effect is exactly zero.
    assert report["command_autocorrelation_rho"] > 0.7
    assert report["uncorrected_hijack_mean_alignment"] > 0.5
    assert abs(report["hijack_mean_alignment"]) < 0.15


def test_block_scaling_reaches_the_classifier(synthetic_run):
    """Scaling before StandardScaler is a no-op; it must be applied after.

    ``StandardScaler`` divides every column by its own standard deviation, so a
    per-block constant applied to the feature matrix is exactly cancelled.  The
    control has to change the fitted model or it cannot fail.
    """
    from efference_probe.probes import BlockScaler

    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 6)) * np.array([1.0, 1.0, 1.0, 50.0, 50.0, 50.0])
    spans = [("a", 3), ("b", 3)]
    standardised = (features - features.mean(0)) / features.std(0)
    scaled = BlockScaler(spans).transform(standardised)
    assert np.allclose(scaled[:, :3], standardised[:, :3] / np.sqrt(3))
    assert not np.allclose(scaled, standardised)
    with pytest.raises(ValueError, match="block spans cover"):
        BlockScaler([("a", 2)]).transform(standardised)


def test_geometric_rotation_delta_does_not_wrap():
    """Componentwise subtraction of absolute axis-angles wraps; composing does not."""
    from efference_probe.datasets import _axis_angle_to_matrix, _matrix_to_axis_angle

    before = np.array([[0.0, 0.0, -3.14159]])
    after = np.array([[0.0, 0.0, 3.14159]])
    naive = float(np.linalg.norm(after - before))
    relative = _axis_angle_to_matrix(after) @ np.transpose(
        _axis_angle_to_matrix(before), (0, 2, 1)
    )
    geometric = float(np.linalg.norm(_matrix_to_axis_angle(relative)))
    assert naive > 6.0
    assert geometric < 1e-4


def test_phase_bias_report_detects_signed_offset():
    """The marginal step-index report cannot see a within-pair signed bias."""
    from efference_probe.datasets import phase_bias_report

    samples = pd.DataFrame(
        {
            "pair_id": np.repeat(np.arange(20), 2),
            "y": np.tile([1, 0], 20),
            "cur_call": np.concatenate([[10 + i % 3, 7 + i % 3] for i in range(20)]),
            "episode_id": np.repeat(np.arange(20), 2),
        }
    )
    report = phase_bias_report(samples)
    assert report["signed_mean_difference"] == pytest.approx(3.0)
    assert report["fraction_negative_earlier"] == 1.0


def test_global_negatives_are_not_treated_as_paired(synthetic_run):
    samples = build_probe_samples(
        synthetic_run.calls,
        warmup_calls=4,
        negatives="global",
        rng=np.random.default_rng(0),
    )
    assert (samples["pair_id"] == -1).all()
    from efference_probe.datasets import phase_bias_report

    assert "note" in phase_bias_report(samples)


def test_raw_positive_count_exceeds_kept_after_pairing(synthetic_run):
    """Pairing drops positives non-randomly; both counts must be available."""
    from efference_probe.datasets import count_raw_positives

    samples = build_probe_samples(
        synthetic_run.calls, warmup_calls=4, rng=np.random.default_rng(0)
    )
    raw = count_raw_positives(synthetic_run.calls, warmup_calls=4)
    kept = int((samples["y"] == 1).sum())
    assert raw >= kept > 0


def test_scheduler_settings_refuse_to_guess(synthetic_run):
    """Class boundaries must come from the run config, never from a default."""
    from efference_probe.analysis import _scheduler_settings

    stripped = RunData(
        run_dir=synthetic_run.run_dir,
        calls=synthetic_run.calls,
        schema=synthetic_run.schema,
        config={"config": {"hijack": {"transform": "swap"}}},
        manifest={},
    )
    with pytest.raises(KeyError, match="min_clean_gap"):
        _scheduler_settings(stripped)


def test_vision_store_rejects_a_missing_call(tmp_path):
    """`payload["vis"][None]` is np.newaxis, not an error -- guard required."""
    budget = DiskBudget(tmp_path, max_gb=1.0, max_hours=1.0)
    writer = RunWriter(tmp_path, budget)
    writer.add_hidden(
        0, 0, np.zeros((1, 1, 4), dtype=np.float16), np.ones(5, dtype=np.float16)
    )
    writer.flush_episode(0)
    store = HiddenStore(tmp_path / "hidden")
    assert store.vision([(0, 0)]).shape == (1, 5)
    with pytest.raises(KeyError):
        store.vision([(0, 7)])


def test_writer_checkpoint_survives_an_interrupted_run(tmp_path):
    """A crash must not cost every label collected so far."""
    budget = DiskBudget(tmp_path, max_gb=1.0, max_hours=1.0)
    writer = RunWriter(tmp_path, budget)
    writer.add_call({"episode_id": 0, "call_idx": 0})
    writer.checkpoint({"stop_reason": "in_progress"})
    assert (tmp_path / "calls.parquet").is_file()
    assert len(pd.read_parquet(tmp_path / "calls.parquet")) == 1
    writer.add_call({"episode_id": 0, "call_idx": 1})
    writer.checkpoint({"stop_reason": "in_progress"})
    assert len(pd.read_parquet(tmp_path / "calls.parquet")) == 2


# --------------------------------------------------------------------------
# Storage and config
# --------------------------------------------------------------------------


def test_writer_round_trip(tmp_path):
    budget = DiskBudget(tmp_path, max_gb=1.0, max_hours=1.0)
    writer = RunWriter(tmp_path, budget)
    hidden = np.random.default_rng(0).normal(size=(2, 3, 16)).astype(np.float16)
    writer.add_hidden(0, 0, hidden)
    writer.add_hidden(0, 1, hidden + 1)
    writer.add_call(
        {"episode_id": 0, "call_idx": 0, "a_cmd_env": flatten(np.zeros((8, 7)))}
    )
    writer.add_call(
        {"episode_id": 0, "call_idx": 1, "a_cmd_env": flatten(np.ones((8, 7)))}
    )
    writer.finalize()

    frame = pd.read_parquet(tmp_path / "calls.parquet")
    assert len(frame) == 2
    assert len(np.asarray(frame["a_cmd_env"].iloc[0])) == 56
    with np.load(tmp_path / "hidden" / "ep00000.npz") as archive:
        assert archive["h"].shape == (2, 2, 3, 16)
        assert list(archive["call_idx"]) == [0, 1]
    assert budget.bytes_written > 0


def test_hidden_store_indexes_by_call(tmp_path):
    budget = DiskBudget(tmp_path, max_gb=1.0, max_hours=1.0)
    writer = RunWriter(tmp_path, budget)
    for call_idx in (3, 4, 9):
        writer.add_hidden(2, call_idx, np.full((1, 1, 4), call_idx, dtype=np.float16))
    writer.flush_episode(2)
    store = HiddenStore(tmp_path / "hidden")
    values = store.vectors([(2, 9), (2, 3)], 0, 0)
    assert np.allclose(values[:, 0], [9.0, 3.0])
    with pytest.raises(KeyError):
        store.vectors([(2, 5)], 0, 0)


@pytest.mark.parametrize(
    "name", ["smoke", "pilot", "main", "main_mirror", "main_freeze"]
)
def test_shipped_configs_validate(name):
    cfg = load_config(str(CONFIG_DIR / f"{name}.yaml"))
    assert cfg.hash()
    assert cfg.model.num_action_chunks * cfg.model.action_dim == 56


def test_config_rejects_bad_values():
    cfg = RunConfig()
    cfg.hijack.transform = "nonsense"
    with pytest.raises(ValueError, match="transform"):
        cfg.validate()

    cfg = RunConfig()
    cfg.hijack.min_clean_gap = 0
    with pytest.raises(ValueError, match="min_clean_gap"):
        cfg.validate()

    cfg = RunConfig()
    cfg.capture.pools = ["nope"]
    with pytest.raises(ValueError, match="pools"):
        cfg.validate()

    cfg = RunConfig()
    cfg.env.max_calls_per_episode = 4
    with pytest.raises(ValueError, match="max_calls_per_episode"):
        cfg.validate()


def test_config_hash_ignores_run_id_but_tracks_settings():
    first, second = RunConfig(run_id="a"), RunConfig(run_id="b")
    assert first.hash() == second.hash()
    second.hijack.p_hijack = 0.5
    assert first.hash() != second.hash()


def test_budget_reports_exhaustion(tmp_path):
    budget = DiskBudget(tmp_path, max_gb=1e-9, max_hours=10.0)
    assert budget.exceeded() is None
    budget.add_bytes(10_000)
    assert budget.exceeded() == "disk"


def test_markdown_table_needs_no_tabulate():
    frame = pd.DataFrame({"name": ["P1", "P2"], "score": [0.512345, float("nan")]})
    rendered = to_markdown(frame)
    lines = rendered.splitlines()
    assert lines[0].startswith("| name")
    assert len(lines) == 4
    assert "0.5123" in rendered
