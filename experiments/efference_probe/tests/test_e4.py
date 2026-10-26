"""Offline tests for the E4 wave tooling (no GPU, LIBERO, or checkpoint).

Run with ``pytest experiments/efference_probe/tests/test_e4.py``.  They pin
the two things E4 promises: the battery is the wave-2 battery unchanged (only
the overrides in ``e4_plan.yaml`` move), and the operator guards wave 2 had to
learn -- never reuse a run dir, completion is a marker not a manifest, ``--jobs``
from the CPU quota, a watchdog that stops our own child -- actually hold.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efference_probe import e4_common as common  # noqa: E402
from efference_probe import e4_report as report  # noqa: E402
from efference_probe import e4_wave as wave_mod  # noqa: E402
from efference_probe.config import load_config  # noqa: E402
from efference_probe.tests.synthetic import make_synthetic_run  # noqa: E402

PLAN = common.load_plan()


def _flat(mapping: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            out.update(_flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


# --------------------------------------------------------------------------
# The plan: unchanged battery, the right run set
# --------------------------------------------------------------------------

# Fields an E4 run may legitimately change relative to its wave-2 config.
BOOKKEEPING = {
    "run_id",
    "notes",
    "out_root",
    "model.model_path",
    "budget.max_gpu_hours",
}
ALLOWED = {
    "s0": {"model.unnorm_key"},
    "s0_envs20": {"model.unnorm_key", "env.num_envs", "env.init_states_per_task"},
    "s1": {"model.unnorm_key", "env.num_envs", "model.do_sample"},
    # Amendment 1's pilot2: the same pilot over all goal tasks x 8 states.
    "s1b": {
        "model.unnorm_key",
        "env.num_envs",
        "model.do_sample",
        "env.task_ids",
        "env.init_states_per_task",
    },
    "s2": {"model.unnorm_key", "env.num_envs", "env.seed"},
}
# Amendment 1: only the sized checkpoint's S2 arms may change episode count.
SIZED = {"env.init_states_per_task"}


@pytest.mark.parametrize("stage", sorted(ALLOWED))
def test_every_planned_run_is_its_wave2_config_plus_listed_overrides(stage):
    for run in common.planned_runs(PLAN, stage):
        sized = stage == "s2" and run.ckpt == PLAN["sizing"]["sized_ckpt"]
        overrides = wave_mod.collect_overrides(
            PLAN,
            run,
            run.run_id,
            f"/ckpt/{run.ckpt}",
            Path("/data"),
            None if stage.startswith("s0") else 10,
            2 * run.expected_s,
            init_states=50 if sized else None,
        )
        resolved = _flat(
            load_config(str(common.CONFIG_DIR / run.config), overrides).to_dict()
        )
        base = _flat(load_config(str(common.CONFIG_DIR / run.config)).to_dict())
        changed = {k for k in resolved if resolved[k] != base[k]} - BOOKKEEPING
        allowed = ALLOWED[stage] | (SIZED if sized else set())
        assert changed <= allowed, (run.run_id, changed - allowed)
        if stage == "s2" and not sized:
            # The reference checkpoint's arms keep the wave-2 episode count.
            assert "env.init_states_per_task" not in changed, run.run_id
        assert resolved["model.unnorm_key"] == "libero_130_no_noops_trajall"
        if stage in ("s1", "s1b", "s2"):
            # Decoding is held at wave 2's sampled protocol for every arm.
            assert resolved["model.do_sample"] is True
            assert resolved["model.temperature"] == pytest.approx(1.6)


def test_s2_is_nine_collections_per_checkpoint_with_the_requested_arms():
    runs = common.planned_runs(PLAN, "s2")
    assert len(runs) == 18
    assert len({r.run_id for r in runs}) == 18
    for ckpt in common.CKPTS:
        mine = [r for r in runs if r.ckpt == ckpt]
        arms = sorted((r.suite, r.transform, r.seed) for r in mine)
        assert arms == sorted(
            [("goal", t, s) for t in ("swap", "mirror", "freeze") for s in (1, 2)]
            + [(suite, "swap", 0) for suite in ("spatial", "object", "long")]
        )
    episodes = {}
    for run in runs:
        cfg = load_config(str(common.CONFIG_DIR / run.config))
        n_tasks = len(cfg.env.task_ids) if cfg.env.task_ids is not None else 10
        episodes[(run.suite, run.transform)] = n_tasks * cfg.env.init_states_per_task
    assert episodes[("goal", "swap")] == 200
    assert episodes[("goal", "mirror")] == episodes[("goal", "freeze")] == 150
    assert episodes[("spatial", "swap")] == episodes[("object", "swap")] == 200
    # suite_long.yaml is kept unchanged at 15 states x 10 tasks; see e4_plan.yaml.
    assert episodes[("long", "swap")] == 150


def test_checkpoints_interleave_so_a_tail_cut_stays_balanced():
    order = [(r.ckpt, r.key) for r in common.planned_runs(PLAN, "s2")]
    assert order[:4] == [
        ("sft", "goal_swap_s1"),
        ("grpo", "goal_swap_s1"),
        ("sft", "goal_swap_s2"),
        ("grpo", "goal_swap_s2"),
    ]


def test_plan_pins_both_checkpoints():
    for ckpt in common.CKPTS:
        spec = PLAN["checkpoints"][ckpt]
        assert len(spec["revision"]) == 40
        assert len(spec["shards"]) == 4
        assert all(len(sha) == 64 for sha in spec["shards"].values())
    assert "lora_adapter/*" in PLAN["checkpoints"]["sft"]["download_exclude"]


# --------------------------------------------------------------------------
# Run ids and completion
# --------------------------------------------------------------------------


def test_run_ids_are_never_reused(tmp_path):
    assert (
        common.next_attempt_id(tmp_path, "e4_sft_goal_swap_s1") == "e4_sft_goal_swap_s1"
    )
    (tmp_path / "e4_sft_goal_swap_s1").mkdir()
    assert (
        common.next_attempt_id(tmp_path, "e4_sft_goal_swap_s1")
        == "e4_sft_goal_swap_s1_r2"
    )
    (tmp_path / "e4_sft_goal_swap_s1_r2").mkdir()
    # A sibling whose name merely starts with the id is not an attempt of it.
    (tmp_path / "e4_sft_goal_swap_s1_extra").mkdir()
    assert common.attempt_ids(tmp_path, "e4_sft_goal_swap_s1") == [
        "e4_sft_goal_swap_s1",
        "e4_sft_goal_swap_s1_r2",
    ]
    assert (
        common.next_attempt_id(tmp_path, "e4_sft_goal_swap_s1")
        == "e4_sft_goal_swap_s1_r3"
    )


def test_completion_is_the_marker_not_the_manifest(tmp_path):
    run_dir = tmp_path / "e4_sft_smoke"
    run_dir.mkdir()
    # The harness's mid-run manifest already says "completed".
    (run_dir / "manifest.json").write_text(json.dumps({"stop_reason": "completed"}))
    assert common.completed_attempt(tmp_path, "e4_sft_smoke") is None
    common.write_json_atomic(
        common.status_path(tmp_path, "e4_sft_smoke", "collect"), {"ok": False}
    )
    assert common.completed_attempt(tmp_path, "e4_sft_smoke") is None
    common.write_json_atomic(
        common.status_path(tmp_path, "e4_sft_smoke", "collect"), {"ok": True}
    )
    assert common.completed_attempt(tmp_path, "e4_sft_smoke") == "e4_sft_smoke"


def test_status_says_incomplete_not_fail_for_a_run_without_a_marker(tmp_path):
    wave = wave_mod.Wave(tmp_path, tmp_path / "logbook.md", PLAN)
    (tmp_path / "e4_sft_goal_swap_s1").mkdir()  # collecting, or cut off
    row = lambda: (
        wave_mod.status_table(wave)
        .set_index("run_id")
        .loc[  # noqa: E731
            "e4_sft_goal_swap_s1", "collect"
        ]
    )
    assert row() == "incomplete"
    common.write_json_atomic(
        common.status_path(tmp_path, "e4_sft_goal_swap_s1", "collect"), {"ok": False}
    )
    assert row() == "FAIL"


def _driver(tmp_path, *argv) -> tuple[int, str]:
    result = subprocess.run(
        [
            sys.executable,
            str(common.PROBE_DIR / "e4_wave.py"),
            "--data-root",
            str(tmp_path),
            "--logbook",
            str(tmp_path / "logbook.md"),
            *argv,
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout + result.stderr


def test_todo_offers_each_job_once_and_never_retries_a_failure(tmp_path):
    rc, out = _driver(tmp_path, "todo")
    todo = json.loads(out)
    assert (
        rc == 0
        and len(todo["collect"]) == 18
        and todo["collect"][0] == ["sft", "goal_swap_s1"]
    )
    done = "e4_sft_goal_swap_s1"
    (tmp_path / done).mkdir()
    common.write_json_atomic(
        common.status_path(tmp_path, done, "collect"), {"ok": True}
    )
    todo = json.loads(_driver(tmp_path, "todo")[1])
    assert ["sft", "goal_swap_s1"] not in todo["collect"] and todo["linear"] == [done]
    common.write_json_atomic(
        common.status_path(tmp_path, done, "probe-linear"), {"ok": True}
    )
    todo = json.loads(_driver(tmp_path, "todo")[1])
    assert todo["linear"] == [] and todo["mlp"] == [done]
    common.write_json_atomic(
        common.status_path(tmp_path, done, "probe-mlp"), {"ok": False}
    )
    assert json.loads(_driver(tmp_path, "todo")[1])["mlp"] == []


def test_todo_queue_puts_the_primary_mlp_rounds_first(tmp_path, capsys):
    # The state the interim wave stopped in: every goal/spatial/object run
    # collected, linear done but GRPO object, MLP done only for SFT goal swap,
    # and an interrupted SFT long attempt.
    wave = wave_mod.Wave(tmp_path, tmp_path / "logbook.md", PLAN)
    runs = common.planned_runs(PLAN, "s2")
    wave.update_state(
        lock={"num_envs": 10, "speed_factor": 1.0, "mujoco_gl": "egl"},
        sizing={"states_per_task": {r.key: 2 * wave_mod.base_states(r) for r in runs}},
    )
    for run in runs:
        (tmp_path / run.run_id).mkdir()
        if run.suite == "long":
            continue
        common.write_json_atomic(
            common.status_path(tmp_path, run.run_id, "collect"), {"ok": True}
        )
        if run.run_id != "e4_grpo_object_swap_s0":
            common.write_json_atomic(
                common.status_path(tmp_path, run.run_id, "probe-linear"), {"ok": True}
            )
    for done in ("e4_sft_goal_swap_s1", "e4_sft_goal_swap_s2"):
        common.write_json_atomic(
            common.status_path(tmp_path, done, "probe-mlp"), {"ok": True}
        )

    assert wave_mod.cmd_todo(wave, None) == 0
    queue = json.loads(capsys.readouterr().out)["queue"]
    assert [i["key"] for i in queue] == [
        "mlp:e4_grpo_goal_swap_s1",
        "mlp:e4_grpo_goal_swap_s2",
        "linear:e4_grpo_object_swap_s0",
        "mlp:e4_sft_goal_mirror_s1",
        "mlp:e4_grpo_goal_mirror_s1",
        "mlp:e4_sft_goal_freeze_s1",
        "mlp:e4_grpo_goal_freeze_s1",
        "mlp:e4_sft_goal_mirror_s2",
        "mlp:e4_grpo_goal_mirror_s2",
        "mlp:e4_sft_goal_freeze_s2",
        "mlp:e4_grpo_goal_freeze_s2",
        "collect:e4_sft_long_swap_s0",
        "collect:e4_grpo_long_swap_s0",
    ]
    by_key = {i["key"]: i for i in queue}
    factor, grace = PLAN["budget"]["watchdog_factor"], PLAN["budget"]["hang_grace_s"]
    long_s = next(r.expected_s for r in runs if r.key == "long_swap_s0")
    assert by_key["mlp:e4_grpo_goal_swap_s1"]["max_s"] == factor * 8400
    assert by_key["mlp:e4_grpo_goal_swap_s1"]["args"] == [
        "probe",
        "--run-id",
        "e4_grpo_goal_swap_s1",
        "--family",
        "mlp",
    ]
    assert by_key["collect:e4_sft_long_swap_s0"]["max_s"] == factor * long_s + grace
    # GRPO's sized arm reserves for twice the episodes.
    assert (
        by_key["collect:e4_grpo_long_swap_s0"]["max_s"] == 2 * factor * long_s + grace
    )
    assert by_key["collect:e4_grpo_long_swap_s0"]["args"] == [
        "collect",
        "--ckpt",
        "grpo",
        "--key",
        "long_swap_s0",
    ]

    wave.halt("two-strike rule")
    capsys.readouterr()
    assert wave_mod.cmd_todo(wave, None) == 0
    queue = json.loads(capsys.readouterr().out)["queue"]
    assert not [i for i in queue if i["kind"] == "collect"]


def test_todo_queue_leaves_out_collections_it_cannot_price(tmp_path, capsys):
    # Before `lock` and `size` the driver refuses to collect anyway.
    wave = wave_mod.Wave(tmp_path, tmp_path / "logbook.md", PLAN)
    assert wave_mod.cmd_todo(wave, None) == 0
    todo = json.loads(capsys.readouterr().out)
    assert len(todo["collect"]) == 18 and todo["queue"] == []


def test_s2_refuses_before_its_gates(tmp_path):
    rc, out = _driver(tmp_path, "s2")
    assert rc == 2 and "S1b (pilot2) gate" in out
    rc, out = _driver(tmp_path, "collect", "--ckpt", "grpo", "--key", "goal_swap_s1")
    assert rc == 2 and "S1b (pilot2) gate" in out


def _s1b_gates(data: Path, yields: dict) -> None:
    for ckpt, y in yields.items():
        common.write_json_atomic(
            data / "e4_gates" / f"s1b_{ckpt}.json",
            {
                "stage": "s1b",
                "run_id": f"e4_{ckpt}_pilot2",
                "passed": True,
                "checks": {},
                "report": {"positives_per_episode": y},
            },
        )


def test_size_matches_samples_with_cap_and_floor(tmp_path):
    _s1b_gates(tmp_path, {"sft": 2.15, "grpo": 0.70})  # r = 3.07
    rc, out = _driver(tmp_path, "size")
    assert rc == 0, out
    sizing = common.read_json(tmp_path / "e4_state.json")["sizing"]
    assert sizing["ratio"] == pytest.approx(2.15 / 0.70)
    states = sizing["states_per_task"]
    assert set(states) == {r.key for r in common.planned_runs(PLAN, "s2")}
    assert states["goal_swap_s1"] == 50  # ceil(20 x 3.07) = 62, capped
    assert states["goal_mirror_s1"] == 50  # ceil(30 x 3.07) = 93, capped
    assert states["long_swap_s0"] == 47  # ceil(15 x 3.07) = 47
    # Frozen once S2 data exists.
    (tmp_path / "e4_sft_goal_swap_s1").mkdir()
    rc, out = _driver(tmp_path, "size")
    assert rc == 2 and "frozen" in out


def test_size_never_shrinks_the_sized_arms(tmp_path):
    _s1b_gates(tmp_path, {"sft": 0.5, "grpo": 2.0})  # r = 0.25
    assert _driver(tmp_path, "size")[0] == 0
    states = common.read_json(tmp_path / "e4_state.json")["sizing"]["states_per_task"]
    assert states["goal_swap_s1"] == 20 and states["long_swap_s0"] == 15


def test_logbook_merge_is_ordered_and_idempotent(tmp_path):
    parts = tmp_path / "parts"
    parts.mkdir()
    (parts / "b.md").write_text(
        "- `2026-10-27 10:00:02` second\n- `2026-10-27 10:00:04` fourth\n"
    )
    (parts / "a.md").write_text(
        "- `2026-10-27 10:00:01` first\n- `2026-10-27 10:00:03` third\n"
    )
    (tmp_path / "logbook.md").write_text("# log\n\n- `2026-10-26 09:00:00` earlier\n")
    for _ in range(2):
        assert _driver(tmp_path, "merge-logbook", "--parts", str(parts))[0] == 0
    lines = (tmp_path / "logbook.md").read_text().splitlines()
    assert lines[2:] == [
        "- `2026-10-26 09:00:00` earlier",
        "- `2026-10-27 10:00:01` first",
        "- `2026-10-27 10:00:02` second",
        "- `2026-10-27 10:00:03` third",
        "- `2026-10-27 10:00:04` fourth",
    ]


# --------------------------------------------------------------------------
# The box
# --------------------------------------------------------------------------


def test_cgroup_quota_parsing(tmp_path):
    (tmp_path / "cpu.max").write_text("2380000 100000\n")
    assert wave_mod.cgroup_cpu_quota(tmp_path) == pytest.approx(23.8)
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert wave_mod.cgroup_cpu_quota(tmp_path) is None
    v1 = tmp_path / "v1"
    (v1 / "cpu").mkdir(parents=True)
    (v1 / "cpu" / "cpu.cfs_quota_us").write_text("1600000")
    (v1 / "cpu" / "cpu.cfs_period_us").write_text("100000")
    assert wave_mod.cgroup_cpu_quota(v1) == pytest.approx(16.0)


def test_usable_cpus_honours_the_quota(tmp_path):
    (tmp_path / "cpu.max").write_text("200000 100000\n")
    assert wave_mod.usable_cpus(tmp_path) <= 2


def test_jobs_is_never_minus_one():
    with pytest.raises(wave_mod.StageError):
        wave_mod.resolve_jobs(-1)
    assert wave_mod.resolve_jobs(None) >= 1
    assert wave_mod.resolve_jobs(None, reserve=10_000) == 1


def test_headroom_is_a_real_write(tmp_path):
    ok, why = wave_mod.headroom_ok(tmp_path, 0.01)
    assert ok, why
    assert not list(tmp_path.glob(".e4_headroom_*"))


def test_watchdog_stops_our_child(tmp_path):
    messages = []
    result = wave_mod.run_child(
        [sys.executable, "-c", "import time; print('up', flush=True); time.sleep(120)"],
        env=dict(__import__("os").environ),
        log_path=tmp_path / "child.log",
        kill_after_s=1.0,
        log=messages.append,
        poll_s=0.2,
    )
    assert result["watchdog_killed"]
    assert result["rc"] != 0
    assert result["elapsed_s"] < 60
    assert b"up" in (tmp_path / "child.log").read_bytes()
    assert any("SIGINT" in m for m in messages)


# --------------------------------------------------------------------------
# Synthetic runs through the real probe pipeline
# --------------------------------------------------------------------------


def _with_manifest(run_dir: Path, success_tasks=(0,)) -> Path:
    """synthetic.py writes no manifest; add one like RunWriter's."""
    calls = pd.read_parquet(run_dir / "calls.parquet")
    episodes = []
    for episode_id, rows in calls.groupby("episode_id"):
        task = int(rows["task_id"].iloc[0])
        episodes.append(
            {
                "episode_id": int(episode_id),
                "task_id": task,
                "n_calls": int(len(rows)),
                "success": task in success_tasks,
                "censored": False,
                "is_probe_episode": bool(int(episode_id) % 5),
            }
        )
    manifest = {
        "n_calls": int(len(calls)),
        "n_episodes": len(episodes),
        "episodes": episodes,
        "stop_reason": "completed",
        "verification": {"passed": True, "max_abs_logprob_diff": 2.3e-7},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return run_dir


@pytest.fixture(scope="module")
def null_run(tmp_path_factory):
    run_dir = make_synthetic_run(
        tmp_path_factory.mktemp("e4") / "e4_sft_pilot",
        n_tasks=3,
        episodes_per_task=10,
        n_calls=24,
        hidden_signal=0.0,
        command_mean=0.4,
        seed=3,
    )
    return _with_manifest(run_dir)


def test_collection_checks_pass_and_catch_a_truncated_archive(null_run, tmp_path):
    checks = wave_mod.collection_checks(null_run, rc=0)
    assert checks["passed"], checks
    broken = tmp_path / "broken"
    subprocess.run(["cp", "-r", str(null_run), str(broken)], check=True)
    next(iter(sorted((broken / "hidden").glob("*.npz")))).write_bytes(b"")
    checks = wave_mod.collection_checks(broken, rc=0)
    assert not checks["passed"]
    assert checks["hidden"]["bad_archives"]


def test_competence_flags_the_task_the_policy_cannot_do(null_run):
    frame = common.competence_report(null_run)
    assert sorted(frame["task_id"]) == [0, 1, 2]
    assert frame.set_index("task_id").loc[0, "success_rate"] == 1.0
    flagged = set(frame.loc[frame["flag_low_success"], "task_id"])
    assert flagged == {1, 2}
    assert (frame["intervention_rel_change"] > 0).all()


@pytest.fixture(scope="module")
def pilot_analysed(null_run):
    probe_script = Path(wave_mod.common.PROBE_DIR) / "run_probes.py"
    result = subprocess.run(
        [
            sys.executable,
            str(probe_script),
            "--run",
            str(null_run),
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
    return null_run


def test_s1_gate_reads_real_run_probes_output(pilot_analysed):
    competence = common.competence_report(pilot_analysed)
    gate = wave_mod.gate_s1(pilot_analysed, {"ok": True}, {"ok": True}, competence)
    checks = gate["checks"]
    assert checks["plumbing_passed_zero_mismatches"]
    assert checks["P0_mean_over_cells_in_band"], gate["report"]["P0"]
    assert checks["step_index_overlap"]
    assert gate["report"]["flagged_tasks"] == [1, 2]
    assert gate["report"]["P2r"]["bacc"] > 0.5


def test_amended_pilot_gate_judges_controls_against_their_own_noise(pilot_analysed):
    """Same pilot, both rules: the strict band reads noise as failure."""
    competence = common.competence_report(pilot_analysed)
    strict = wave_mod.gate_s1(pilot_analysed, {"ok": True}, {"ok": True}, competence)
    amended = wave_mod.gate_s1(
        pilot_analysed, {"ok": True}, {"ok": True}, competence, "amended", "s1b"
    )
    assert amended["stage"] == "s1b" and amended["rule"] == "amended"
    for name in ("C_cmd", "C_phase"):
        cell = amended["report"][name]
        expected = abs(cell["bacc"] - 0.5) <= 2 * cell["sd"]
        assert amended["checks"][f"{name}_within_2sd_of_chance"] == expected
        assert f"{name}_in_band" in strict["checks"]
    report_ = amended["report"]
    assert amended["checks"]["min_samples_per_class"] == (
        min(report_["n_positive"], report_["n_negative"]) >= 40
    )
    assert report_["positives_per_episode"] == pytest.approx(
        report_["n_positive"] / report_["n_episodes"]
    )


def test_report_reads_a_real_analysis_directory(pilot_analysed):
    row = report.read_run(pilot_analysed, "Base-Lora", "goal", "swap", 1)
    assert row.source == "analysis"
    assert row.families["linear"]["floor"] is not None
    assert row.n and row.n > 0
    assert row.e3["n_ep"]
    lines = report.table_rows([row], "linear")
    assert len(lines) == 3 and "[TO VERIFY" not in lines[2].split("|")[5]


def _fake_checkpoint(path: Path, ckpt: str) -> str:
    """Files ``verify_local --no-hash`` looks for; the fake collector never reads them."""
    path.mkdir(parents=True)
    shards = PLAN["checkpoints"][ckpt]["shards"]
    (path / "dataset_statistics.json").write_text(json.dumps({PLAN["unnorm_key"]: {}}))
    (path / "config.json").write_text("{}")
    weight_map = {f"w{i}": name for i, name in enumerate(shards)}
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    for name in shards:
        (path / name).write_bytes(b"")
    return str(path)


def test_stage_flow_end_to_end_with_a_fake_collector(tmp_path, monkeypatch):
    """init -> s0 -> lock -> s1 -> s1b -> size -> s2 -> s3 -> report, then failures.

    The collector is ``tests/fake_collect.py`` (synthetic runs); everything
    after it -- gates, run_probes.py, markers, ledger, report -- is real.
    """
    import yaml

    monkeypatch.setenv(
        "E4_COLLECT_SCRIPT", str(Path(__file__).with_name("fake_collect.py"))
    )
    plan = json.loads(json.dumps(PLAN))
    plan["probes"]["linear"]["args"] = [
        "--stage",
        "main",
        "--permutations",
        "2",
        "--fast",
    ]
    plan["probes"]["mlp"]["args"] = [
        "--stage",
        "main",
        "--probe-family",
        "mlp",
        "--fast",
        "--no-controls",
        "--probes",
        "P0",
        "--layers",
        "0",
        "--pools",
        "ctx_mean",
        "--no-figures",
    ]
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False))
    data = tmp_path / "data"

    def run(*argv):
        return _driver(data, "--plan", str(plan_path), *argv)

    sft = _fake_checkpoint(tmp_path / "ck" / "sft", "sft")
    grpo = _fake_checkpoint(tmp_path / "ck" / "grpo", "grpo")
    assert (
        run("init", "--sft", sft, "--grpo", grpo, "--no-hash", "--headroom-gb", "0.01")[
            0
        ]
        == 0
    )
    rc, out = run("s0")
    assert rc == 0, out
    assert run("lock", "--num-envs", "10")[0] == 0
    run("s1")  # the registered pilot; its gate files stay as the record
    run("s1b")
    gate_paths = [data / "e4_gates" / f"s1b_{c}.json" for c in common.CKPTS]
    assert all((data / "e4_gates" / f"s1_{c}.json").is_file() for c in common.CKPTS)
    if not all(common.read_json(p)["passed"] for p in gate_paths):
        rc, out = run("s2")
        assert rc == 2 and "S1b (pilot2) gate" in out
        for path in gate_paths:  # gate statistics are not under test here
            common.write_json_atomic(path, {**common.read_json(path), "passed": True})
    # The sized checkpoint cannot collect S2 before the sizing is frozen.
    rc, out = run("collect", "--ckpt", "grpo", "--key", "goal_swap_s1")
    assert rc == 2 and "not sized" in out
    rc, out = run("size")
    assert rc == 0, out
    states = common.read_json(data / "e4_state.json")["sizing"]["states_per_task"]
    rc, out = run("s2", "--only", "goal_swap_s1", "--probes", "inline")
    assert rc == 0, out
    for ckpt in common.CKPTS:
        run_id = f"e4_{ckpt}_goal_swap_s1"
        assert common.job_status(data, run_id, "probe-linear")["ok"]
        assert (data / "e4_derived" / run_id / "competence.csv").is_file()
    # The sized arm's watchdog expectation follows its episode count.
    grpo = common.job_status(data, "e4_grpo_goal_swap_s1", "collect")
    assert grpo["expected_s"] == pytest.approx(1408 * states["goal_swap_s1"] / 20)
    rc, out = run("s3")
    assert rc == 0, out
    assert common.job_status(data, "e4_grpo_goal_swap_s1", "probe-mlp")["ok"]
    text = report.render(plan, data, None)
    row = next(
        l
        for l in text.splitlines()
        if "`e4_grpo_goal_swap_s1`" in l and l.startswith("|")
    )
    assert "[TO VERIFY" not in row.split("|")[5]  # the floor was read
    assert common.gpu_hours_used(data) > 0

    # A collection that dies: halt, refuse while halted, new id on retry,
    # two strikes and the lane stays down.
    monkeypatch.setenv("FAKE_COLLECT_FAIL", "1")
    assert run("collect", "--ckpt", "sft", "--key", "goal_swap_s2")[0] == 1
    assert (data / "E4_HALT").exists()
    assert run("collect", "--ckpt", "sft", "--key", "goal_swap_s2")[0] == 2
    assert run("resume", "--reason", "test")[0] == 0
    assert run("collect", "--ckpt", "sft", "--key", "goal_swap_s2")[0] == 1
    assert (data / "e4_sft_goal_swap_s2_r2").is_dir()
    assert run("resume", "--reason", "test")[0] == 0
    rc, out = run("collect", "--ckpt", "sft", "--key", "goal_swap_s2")
    assert rc == 2 and "two-strike" in out
    assert not (data / "e4_sft_goal_swap_s2_r3").exists()


# --------------------------------------------------------------------------
# The pre-registered rules
# --------------------------------------------------------------------------


def _wave2(run_id: str) -> report.RunNumbers:
    entry = next(e for e in PLAN["existing"]["runs"] if e["run_id"] == run_id)
    return report.read_wave2_report(
        run_id, entry["label"], entry["suite"], entry["transform"], entry["seed"]
    )


def test_wave2_fallback_parses_the_committed_report():
    row = _wave2("main02")
    linear, mlp = row.families["linear"], row.families["mlp"]
    assert linear["floor"] == pytest.approx(0.5413)
    assert linear["perm_p95"] == pytest.approx(0.5233)
    assert linear["P1"]["bacc"] == pytest.approx(0.543)
    assert linear["P2r"]["bacc"] == pytest.approx(0.711)
    assert mlp["floor"] == pytest.approx(0.5320)
    assert row.e3["p_logprob"] == pytest.approx(0.4433)
    assert row.success["clean"] == pytest.approx(0.514)
    assert row.n == 1266


def test_delta_reproduces_every_published_wave2_verdict():
    """δ=0.04 was calibrated on the paper's own wave-2 verdicts before E4 data.

    If this fails, the rule changed -- which the pre-registration forbids.
    """
    expect = {
        ("main02", "linear"): ("at", []),
        ("main02", "mlp"): ("at", []),
        ("main02_mirror", "linear"): ("above", []),
        ("main02_mirror", "mlp"): ("above", []),
        ("main02_freeze", "mlp"): ("above", []),
        ("suite_spatial02", "linear"): ("at", []),
        ("suite_spatial02", "mlp"): ("at", []),
        ("suite_object01", "linear"): ("at", []),
        ("suite_long01", "linear"): ("at", []),
    }
    for (run_id, family), (status, beyond) in expect.items():
        verdict = report.ladder_verdict(_wave2(run_id), family)
        assert verdict["status"] == status, (run_id, family, verdict)
        if status == "above":
            # Every gross-violation excursion is bounded by C_dstates (paper T5).
            assert verdict["beyond_observables"] == beyond, (run_id, family, verdict)


def _numbers(floor, rungs, p2r=0.8, dstates=0.5, cmd=0.5, label="GRPO", seed=1):
    cell = lambda b: {
        "bacc": b,
        "sd": 0.02,
        "auroc": b,
        "cell": "L0/ctx_mean",
        "n_cells": 27,
    }  # noqa: E731
    family = {
        "floor": floor,
        "P2r": cell(p2r),
        "C_dstates": cell(dstates),
        "C_cmd": cell(cmd),
    }
    family.update({r: cell(b) for r, b in zip(report.RUNGS, rungs)})
    return report.RunNumbers(
        label, f"x{seed}", "goal", "swap", seed, "analysis", families={"mlp": family}
    )


def test_checkpoint_verdict_needs_both_seeds_and_the_observables_twin():
    above = [_numbers(0.53, (0.60, 0.55, 0.55), seed=s) for s in (1, 2)]
    assert report.checkpoint_verdict(above, "mlp") == "above floor"
    at = [_numbers(0.53, (0.55, 0.54, 0.56), seed=s) for s in (1, 2)]
    assert report.checkpoint_verdict(at, "mlp") == "at floor"
    split = [above[0], at[1]]
    assert report.checkpoint_verdict(split, "mlp").startswith("inconclusive")
    perceived = [
        _numbers(0.53, (0.60, 0.55, 0.55), dstates=0.70, seed=s) for s in (1, 2)
    ]
    assert "C_dstates" in report.checkpoint_verdict(perceived, "mlp")
    # R2's other arm: rungs clear C_dstates + δ but C_cmd is off chance, and the
    # label must name C_cmd rather than blame C_dstates.
    off_chance = [_numbers(0.53, (0.60, 0.55, 0.55), cmd=0.552, seed=s) for s in (1, 2)]
    verdict = report.checkpoint_verdict(off_chance, "mlp")
    assert "C_cmd off chance" in verdict and "C_dstates" not in verdict
    label = report.explained_label(report.ladder_verdict(off_chance[0], "mlp"))
    assert label == (
        "none (C_cmd 0.552 outside [0.45, 0.55]; rungs clear C_dstates 0.500 + δ)"
    )
    both = report.ladder_verdict(
        _numbers(0.53, (0.60, 0.55, 0.55), dstates=0.70, cmd=0.40), "mlp"
    )
    assert both["explained_by"] == ["C_dstates", "C_cmd"]
    weak = [_numbers(0.53, (0.55, 0.54, 0.54), p2r=0.58, seed=s) for s in (1, 2)]
    assert "uninformative" in report.checkpoint_verdict(weak, "mlp")
    assert report.checkpoint_verdict([above[0], None], "mlp").startswith("inconclusive")


def _e4_rows(grpo_rungs, e1=0.6):
    """A complete set of E4 goal rows; Base-Lora always at floor."""
    rows = []
    for label, rungs in (("Base-Lora", (0.55, 0.55, 0.55)), ("GRPO", grpo_rungs)):
        for transform in ("swap", "mirror", "freeze"):
            for seed in (1, 2):
                row = _numbers(0.53, rungs, label=label, seed=seed)
                row.transform = transform
                row.e1 = {"r2": e1, "cell": "L28/act_mean"}
                surprised = label == "GRPO" and transform == "freeze"
                row.e3 = {
                    "d_entropy": 0.02 if surprised else 0.0,
                    "p_entropy": 0.01 if surprised else 0.6,
                    "d_logprob": -1.0,
                    "p_logprob": 0.02 if surprised else 0.6,
                    "n_ep": 100,
                }
                gap = 0.05 if label == "GRPO" else 0.15
                row.success = {"overall": 0.5, "probe": 0.5 - gap, "clean": 0.5}
                rows.append(row)
    return rows


def test_full_readout_applies_r4_to_r9():
    text = "\n".join(report.readout(PLAN, _e4_rows((0.60, 0.55, 0.55))))
    assert "Base-Lora: at floor; GRPO: above floor. Contrast: (a)" in text
    assert "**R6 (mirror/freeze, MLP).** supports (a)" in text
    assert "**R7 (E3 migration).** migrated toward mismatch-gated" in text
    assert "**R8 (compensation).** strengthens" in text
    withheld = "\n".join(report.readout(PLAN, _e4_rows((0.60, 0.55, 0.55), e1=0.2)))
    assert "withheld under R9" in withheld
    assert "Contrast: (e)" in withheld


def test_contrast_maps_onto_the_paper_preregistration():
    assert report.contrast("at floor", "above floor").startswith("(a)")
    assert report.contrast("at floor", "at floor").startswith("(b)")
    assert report.contrast("inconclusive (x)", "at floor").startswith("(e)")


def test_report_renders_with_nothing_collected(tmp_path):
    text = report.render(PLAN, tmp_path, None)
    assert "## libero_goal / swap / mlp" in text
    assert "## libero_long / swap / mlp" not in text  # S3 is goal-only
    assert "not collected" in text
    assert "goal-traj1†" in text
    # No number for an E4 row can appear before E4 data exists.
    for line in text.splitlines():
        if "`e4_" in line and line.startswith("|"):
            assert "[TO VERIFY" in line


def test_modal_launcher_imports_inside_the_container():
    # Modal imports the launcher as /root/modal_e4.py in every container.
    pytest.importorskip("modal")
    from efference_probe import modal_e4

    assert modal_e4._repo_local("/root/modal_e4.py", local=False) == Path(modal_e4.REPO)
    assert modal_e4.REPO_LOCAL == Path(__file__).resolve().parents[2].parent


def test_modal_image_pins_a_pyarrow_that_imports_under_numpy_1():
    # The image's numpy is 1.26.4; pyarrow >= 20 refuses to import under it.
    pytest.importorskip("modal")
    from efference_probe import modal_e4

    assert "pyarrow<20" in modal_e4.ANALYSIS_PINS
    assert not any(
        p.startswith("pyarrow") and p != "pyarrow<20" for p in modal_e4.ANALYSIS_PINS
    )


def test_modal_image_pin_survives_the_build_shell():
    # run_commands goes through /bin/sh, where a bare `pyarrow<20` is a redirect.
    pytest.importorskip("modal")
    import shlex

    from efference_probe import modal_e4

    quoted = " ".join(map(shlex.quote, modal_e4.ANALYSIS_PINS))
    assert shlex.split(quoted) == list(modal_e4.ANALYSIS_PINS)
    assert "'pyarrow<20'" in quoted


def test_modal_gpu_limit_clears_the_measured_pilot_rss():
    # e4_*_pilot2 peaked at 62.51 GB tree RSS; the old 64 GiB limit left ~6 GB.
    pytest.importorskip("modal")
    from efference_probe import modal_e4

    request_mib, limit_mib = modal_e4.GPU_MEMORY
    assert request_mib == 49152
    assert limit_mib * 2**20 >= 1.4 * 62.51e9


def test_modal_orchestrator_does_not_read_a_running_call_as_finished():
    # modal 1.6 raises the builtin TimeoutError from get(timeout=0) while a
    # call runs; the first live wave exited at once because of it.
    pytest.importorskip("modal")
    import modal

    from efference_probe import modal_e4

    class Call:
        def __init__(self, error=None):
            self.error = error

        def get(self, timeout=None):
            if self.error is not None:
                raise self.error
            return 0

    assert modal_e4._outcome(Call(TimeoutError())) is None
    assert modal_e4._outcome(Call(modal.exception.TimeoutError())) is None
    assert modal_e4._outcome(Call()) == "rc=0"
    assert modal_e4._outcome(Call(RuntimeError("x"))) == "error: RuntimeError"
    assert modal_e4._outcome(Call(modal.exception.OutputExpiredError())) == "expired"


def _round(key, kind, max_s=3600.0):
    return {"key": key, "kind": kind, "args": [], "max_s": max_s, "rank": 0}


def test_modal_orchestrator_never_starts_a_round_already_in_flight():
    # A second probe on a run renames its live analysis dir to .partial_*.
    pytest.importorskip("modal")
    from collections import Counter

    from efference_probe import modal_e4

    queue = [_round("linear:a", "linear"), _round("linear:b", "linear")]
    caps = {"linear": 2}
    chosen, held = modal_e4.pick(queue, {"linear:a"}, Counter(), 0.0, 1e9, caps)
    assert [i["key"] for i in chosen] == ["linear:b"] and held is None
    chosen, _ = modal_e4.pick(queue, {"linear:a", "linear:b"}, Counter(), 0, 1e9, caps)
    assert chosen == []


def test_modal_orchestrator_caps_each_kind_and_holds_at_the_budget():
    # The interim wave started all ten MLP rounds at once and lost every one
    # of them to the workspace spend limit.
    pytest.importorskip("modal")
    from collections import Counter

    from efference_probe import modal_e4

    queue = [
        _round("mlp:grpo_swap_s1", "mlp", 16800),
        _round("mlp:grpo_swap_s2", "mlp", 16800),
        _round("linear:grpo_object", "linear", 1600),
        _round("mlp:sft_mirror_s1", "mlp", 16800),
        _round("collect:sft_long", "collect", 4596),
    ]
    caps = dict(modal_e4.DEFAULT_MAX_RUNNING)
    chosen, held = modal_e4.pick(queue, set(), Counter(), 0.0, 1e9, caps)
    assert [i["key"] for i in chosen] == [
        "mlp:grpo_swap_s1",
        "mlp:grpo_swap_s2",
        "linear:grpo_object",
        "collect:sft_long",
    ]
    assert held is None
    # Two MLP rounds already running: no third, whatever the budget.
    chosen, _ = modal_e4.pick(queue, set(), Counter(mlp=2), 0.0, 1e9, caps)
    assert all(i["kind"] != "mlp" for i in chosen)
    # Room for one MLP reservation: the second is held, and so is every round
    # queued after it, even the cheap linear one.
    one = modal_e4.reserve_usd(queue[0])
    chosen, held = modal_e4.pick(queue, set(), Counter(), 0.0, one * 1.5, caps)
    assert [i["key"] for i in chosen] == ["mlp:grpo_swap_s1"]
    assert held["key"] == "mlp:grpo_swap_s2"
    # What is already committed counts against the budget.
    chosen, held = modal_e4.pick(queue, set(), Counter(), one, one * 1.5, caps)
    assert chosen == [] and held["key"] == "mlp:grpo_swap_s1"
    # --no-lane: collections are never started here.
    chosen, _ = modal_e4.pick(queue, set(), Counter(), 0.0, 1e9, {**caps, "collect": 0})
    assert "collect:sft_long" not in [i["key"] for i in chosen]


def test_modal_journal_counts_finished_running_and_overrun_rounds():
    pytest.importorskip("modal")
    from efference_probe import modal_e4

    def start(key, rate, reserve_s):
        return {
            "event": "start",
            "key": key,
            "t": 0.0,
            "usd_per_h": rate,
            "reserve_s": reserve_s,
        }

    entries = [
        start("done", 1.0, 7200),
        {"event": "end", "key": "done", "t": 3600.0, "outcome": "rc=0"},
        start("running", 2.0, 3600),
    ]
    # done: 1 h at $1; running: 0.5 h at $2 so far, $2 reserved in all.
    assert modal_e4.journal_usd(entries, now=1800.0) == pytest.approx((2.0, 3.0))
    # A round past its reservation is counted at what it has actually run.
    assert modal_e4.journal_usd(entries, now=7200.0) == pytest.approx((5.0, 5.0))
    # A round cut off with no end event keeps counting: the journal over-states.
    assert modal_e4.journal_usd(entries[:1], now=7200.0) == pytest.approx((2.0, 2.0))


def test_modal_rates_price_containers_at_their_limits():
    pytest.importorskip("modal")
    from efference_probe import modal_e4

    cpu, mem = modal_e4.PROBE_CPU[1], modal_e4.PROBE_MEMORY[1] / 1024
    assert modal_e4.USD_PER_H["mlp"] == pytest.approx(
        3600 * (cpu * 0.0000131 + mem * 0.00000222)
    )
    if modal_e4.GPU == "L40S":
        assert modal_e4.USD_PER_H["collect"] == pytest.approx(3.284, abs=0.001)
    # The slowest completed MLP round (6720 s) fits its reservation twice over.
    mlp = _round("mlp:x", "mlp", 2 * PLAN["probes"]["mlp"]["expected_s"])
    assert modal_e4.reserve_usd(mlp) > 2 * modal_e4.USD_PER_H["mlp"] * 6720 / 3600


def test_modal_logbook_part_label_is_a_flat_file_name():
    # The init part landed in data/e4_logbook/<stamp>_cpu_init_sft_/vol/ckpts/sft.md.
    pytest.importorskip("modal")
    from efference_probe import modal_e4

    label = modal_e4._part_label("cpu", ["init", "--sft", "/vol/ckpts/sft"])
    assert "/" not in label
    assert label == "cpu_init_sft_vol_ckpts_sft"
    assert (
        modal_e4._part_label("cpu", ["probe", "--run-id", "e4_x"])
        == "cpu_probe_run-id_e4_x"
    )
