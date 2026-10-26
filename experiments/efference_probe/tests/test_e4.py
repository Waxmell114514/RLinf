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
    "s2": {"model.unnorm_key", "env.num_envs", "env.seed"},
}


@pytest.mark.parametrize("stage", sorted(ALLOWED))
def test_every_planned_run_is_its_wave2_config_plus_listed_overrides(stage):
    for run in common.planned_runs(PLAN, stage):
        overrides = wave_mod.collect_overrides(
            PLAN,
            run,
            run.run_id,
            f"/ckpt/{run.ckpt}",
            Path("/data"),
            None if stage.startswith("s0") else 10,
            2 * run.expected_s,
        )
        resolved = _flat(
            load_config(str(common.CONFIG_DIR / run.config), overrides).to_dict()
        )
        base = _flat(load_config(str(common.CONFIG_DIR / run.config)).to_dict())
        changed = {k for k in resolved if resolved[k] != base[k]} - BOOKKEEPING
        assert changed <= ALLOWED[stage], (run.run_id, changed - ALLOWED[stage])
        assert resolved["model.unnorm_key"] == "libero_130_no_noops_trajall"
        if stage in ("s1", "s2"):
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


def test_s2_refuses_before_its_gates(tmp_path):
    rc, out = _driver(tmp_path, "s2")
    assert rc == 2 and "S1 gate" in out
    rc, out = _driver(tmp_path, "collect", "--ckpt", "grpo", "--key", "goal_swap_s1")
    assert rc == 2 and "S1 gate" in out


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
    """init -> s0 -> lock -> s1 -> s2 -> s3 -> report, then the failure paths.

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
    run("s1")
    gate_paths = [data / "e4_gates" / f"s1_{c}.json" for c in common.CKPTS]
    if not all(common.read_json(p)["passed"] for p in gate_paths):
        rc, out = run("s2")
        assert rc == 2 and "S1 gate" in out
        for path in gate_paths:  # gate statistics are not under test here
            common.write_json_atomic(path, {**common.read_json(path), "passed": True})
    rc, out = run("s2", "--only", "goal_swap_s1", "--probes", "inline")
    assert rc == 0, out
    for ckpt in common.CKPTS:
        run_id = f"e4_{ckpt}_goal_swap_s1"
        assert common.job_status(data, run_id, "probe-linear")["ok"]
        assert (data / "e4_derived" / run_id / "competence.csv").is_file()
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
