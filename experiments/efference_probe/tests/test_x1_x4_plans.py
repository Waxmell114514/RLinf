"""X1 and X4 plans on the unchanged E4 driver (PREREG_X1_X4.md §4).

Same contract as E4's plan tests: each planned run is its wave-2 config plus
only the overrides the plan lists, and E4's own plan behaves exactly as before.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efference_probe import e4_common as common  # noqa: E402
from efference_probe import e4_wave as wave_mod  # noqa: E402
from efference_probe.config import load_config  # noqa: E402
from efference_probe.monitor import LinearMonitor  # noqa: E402
from efference_probe.tests.test_e4 import BOOKKEEPING, _flat  # noqa: E402

X1 = common.load_plan(common.PROBE_DIR / "x1_plan.yaml")
X4 = common.load_plan(common.PROBE_DIR / "x4_plan.yaml")
E4 = common.load_plan()

X1_EXTRA = {"capture.capture_vision", "capture.extra_observables"}
X4_EXTRA = {
    "monitor.enabled",
    "monitor.artifact",
    "monitor.artifact_sha256",
    "capture.layers",
    "capture.pools",
}
ALLOWED = {
    ("x1", "s0"): X1_EXTRA | {"env.init_states_per_task", "model.do_sample"},
    ("x1", "s1b"): X1_EXTRA
    | {
        "env.num_envs",
        "model.do_sample",
        "env.task_ids",
        "env.init_states_per_task",
    },
    ("x1", "s2"): X1_EXTRA | {"env.num_envs", "env.seed"},
    ("x4", "s0"): X4_EXTRA
    | {
        "model.unnorm_key",
        "env.task_ids",
        "env.init_states_per_task",
        "env.num_envs",
        "monitor.assignment",
    },
    ("x4", "s2"): X4_EXTRA
    | {"model.unnorm_key", "env.num_envs", "env.seed", "env.init_states_per_task"},
}


def _resolved(plan, run, sized):
    overrides = wave_mod.collect_overrides(
        plan,
        run,
        run.run_id,
        f"/ckpt/{run.ckpt}",
        Path("/data"),
        None if run.stage == "s0" else 10,
        2 * run.expected_s,
        init_states=50 if sized else None,
    )
    resolved = _flat(
        load_config(str(common.CONFIG_DIR / run.config), overrides).to_dict()
    )
    base = _flat(load_config(str(common.CONFIG_DIR / run.config)).to_dict())
    return resolved, {k for k in resolved if resolved[k] != base[k]} - BOOKKEEPING


@pytest.mark.parametrize("wave, stage", sorted(ALLOWED))
def test_every_planned_x_run_is_its_wave2_config_plus_listed_overrides(wave, stage):
    plan = X1 if wave == "x1" else X4
    sizing = plan.get("sizing")
    for run in common.planned_runs(plan, stage):
        assert run.run_id == f"{wave}_{run.ckpt}_{run.key}"
        sized = bool(sizing) and stage == "s2" and run.ckpt == sizing["sized_ckpt"]
        resolved, changed = _resolved(plan, run, sized)
        allowed = ALLOWED[(wave, stage)] | (
            {"env.init_states_per_task"} if sized else set()
        )
        assert changed <= allowed, (run.run_id, changed - allowed)
        assert resolved["model.unnorm_key"] == plan["unnorm_key"]
        # Every X1/X4 collection, smokes included, samples (PREREG §1).
        assert resolved["model.do_sample"] is True
        assert resolved["model.temperature"] == pytest.approx(1.6)
        if wave == "x1":
            assert resolved["capture.extra_observables"] is True
            assert resolved["capture.capture_vision"] is True
            if stage == "s2" and not sized:
                assert "env.init_states_per_task" not in changed
        else:
            assert resolved["monitor.enabled"] is True
            assert resolved["hijack.transform"] == "mirror"
            if stage == "s2":
                assert resolved["env.init_states_per_task"] == 50
                assert resolved["env.task_ids"] == [2, 3, 6, 8, 9]
            else:
                # The smoke runs S2's locked env count (Amendment 1).
                assert resolved["env.num_envs"] == 10


def test_x4_monitors_are_the_frozen_artifacts():
    for ckpt in common.CKPTS:
        run = common.planned_runs(X4, "s2")[0]
        run = common.find_planned(X4, ckpt, run.key)
        resolved, _ = _resolved(X4, run, sized=False)
        artifact = common.REPO_ROOT / resolved["monitor.artifact"]
        monitor = LinearMonitor.load(
            artifact, expect_sha256=resolved["monitor.artifact_sha256"]
        )
        assert monitor.meta["ckpt"] == ckpt
        assert monitor.meta["gate_passed"] is True


def test_x4_runs_grpo_first_and_x1_runs_sft_first():
    assert [r.ckpt for r in common.planned_runs(X4, "s2")][:2] == ["grpo", "sft"]
    assert [r.ckpt for r in common.planned_runs(X1, "s2")][:2] == ["sft", "grpo"]
    assert [r.key for r in common.planned_runs(X1, "s2")][::2] == [
        "swap",
        "mirror",
        "freeze",
    ]


def test_x_registration_markers_are_in_the_logbook_and_e4_keeps_its_own():
    assert wave_mod._missing_markers(X1) == []
    assert wave_mod._missing_markers(X4) == []
    assert wave_mod._missing_markers(E4) == []
    assert "prereg_markers" not in E4
    assert [r.run_id for r in common.planned_runs(E4, "s2")][:2] == [
        "e4_sft_goal_swap_s1",
        "e4_grpo_goal_swap_s1",
    ]


def test_s2_prerequisites_follow_the_plan(tmp_path):
    for plan, needed in (
        (X4, "S0 gate"),
        (X1, r"S1b \(pilot2\) gate"),
        (E4, r"S1b \(pilot2\) gate"),
    ):
        wave = wave_mod.Wave(tmp_path / plan["wave"], tmp_path / "log.md", plan)
        wave.data_root.mkdir()
        with pytest.raises(wave_mod.StageError, match=needed):
            wave_mod.require_s2_ready(wave)


def test_extension_checks_are_empty_for_runs_with_both_flags_off(tmp_path):
    assert wave_mod.extension_checks(tmp_path, None, smoke=True) == {}


def test_launcher_routes_later_waves_to_their_plan_and_data_root():
    modal_e4 = pytest.importorskip("efference_probe.modal_e4")
    assert modal_e4.wave_args("e4", ["s0"]) == ["s0"]
    routed = modal_e4.wave_args(
        "x4", ["collect", "--ckpt", "grpo", "--key", "mirror_s3"]
    )
    assert routed[:2] == ["--data-root", f"{modal_e4.DATA}/x4"]
    assert routed[2] == "--plan" and routed[3].endswith("/x4_plan.yaml")
    assert modal_e4._command_args(routed) == [
        "collect",
        "--ckpt",
        "grpo",
        "--key",
        "mirror_s3",
    ]
    assert modal_e4._part_label("gpu", routed) == "gpu_collect_ckpt_grpo"
    # The driver's parser takes the routed global options, the last one winning.
    args = wave_mod.build_parser().parse_args(
        ["--data-root", "/vol/data", "--logbook", "/l.md", *routed]
    )
    assert args.data_root == f"{modal_e4.DATA}/x4" and args.plan.endswith(
        "x4_plan.yaml"
    )
    assert args.command == "collect"


def _pass_s0(wave):
    for ckpt in common.CKPTS:
        common.write_json_atomic(
            wave.gate_path("s0", ckpt),
            {
                "passed": True,
                "mujoco_gl": "egl",
                "run_id": f"smoke_{ckpt}",
                "checks": {},
            },
        )


@pytest.mark.parametrize("plan", [X1, X4], ids=["x1", "x4"])
def test_lock_and_status_run_on_plans_without_every_stage(tmp_path, plan, capsys):
    """Review F1: both crashed on plans lacking s1 (and s1b), blocking S2."""
    wave = wave_mod.Wave(tmp_path / plan["wave"], tmp_path / "log.md", plan)
    wave.data_root.mkdir()
    _pass_s0(wave)
    args = wave_mod.build_parser().parse_args(["lock", "--num-envs", "10"])
    assert wave_mod.cmd_lock(wave, args) == 0
    assert wave.lock["num_envs"] == 10
    assert wave_mod.cmd_status(wave, args) == 0
    assert common.planned_runs(plan, "s1") == []


def test_x4_s2_needs_only_its_s0_gates(tmp_path):
    wave = wave_mod.Wave(tmp_path / "x4", tmp_path / "log.md", X4)
    wave.data_root.mkdir()
    _pass_s0(wave)
    wave_mod.require_s2_ready(wave)  # no S1b, no sizing: S0 is enough


def test_stage_reserve_covers_each_commands_worst_case():
    modal_e4 = pytest.importorskip("efference_probe.modal_e4")
    budget = X1["budget"]
    grpo_swap = common.find_planned(X1, "grpo", "swap")
    sized = (
        budget["watchdog_factor"]
        * grpo_swap.expected_s
        * X1["sizing"]["max_states_per_task"]
        / wave_mod.base_states(grpo_swap)
        + budget["hang_grace_s"]
    )
    assert modal_e4.stage_reserve_s(
        X1, ["collect", "--ckpt", "grpo", "--key", "swap"]
    ) == pytest.approx(sized)
    sft_swap = common.find_planned(X1, "sft", "swap")
    assert modal_e4.stage_reserve_s(
        X1, ["collect", "--ckpt", "sft", "--key", "swap"]
    ) == pytest.approx(
        budget["watchdog_factor"] * sft_swap.expected_s + budget["hang_grace_s"]
    )
    s0 = modal_e4.stage_reserve_s(X4, ["s0"])
    assert s0 == pytest.approx(
        2 * (X4["budget"]["watchdog_factor"] * 300 + X4["budget"]["hang_grace_s"])
    )
    assert modal_e4.stage_reserve_s(X1, ["size"]) == modal_e4.CPU_COMMAND_RESERVE_S


def test_stage_gate_holds_a_call_that_does_not_fit_the_cap():
    modal_e4 = pytest.importorskip("efference_probe.modal_e4")
    rate = 3.6  # $/h, so one hour of reservation costs $3.60
    journal = [
        {"event": "start", "key": "a", "t": 0.0, "usd_per_h": rate, "reserve_s": 3600},
        {"event": "end", "key": "a", "t": 3600.0},
    ]
    fits, committed, cost = modal_e4.stage_gate(
        journal, 7200.0, 8.0, rate, 3600 - modal_e4.STARTUP_S
    )
    assert committed == pytest.approx(3.6) and cost == pytest.approx(3.6) and fits
    fits, _, _ = modal_e4.stage_gate(
        journal, 7200.0, 7.0, rate, 3600 - modal_e4.STARTUP_S
    )
    assert not fits
    running = journal[:1]  # no end entry: its full reservation stays committed
    _, committed, _ = modal_e4.stage_gate(running, 60.0, 8.0, rate, 0)
    assert committed == pytest.approx(3.6)
