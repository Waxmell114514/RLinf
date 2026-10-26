#!/usr/bin/env python
"""E4 wave driver: the unchanged hijack->probe battery on SFT vs GRPO.

Every stage stops at its gate; running the next command is the decision to
continue.  Typical order on one GPU box::

    P=experiments/efference_probe
    python $P/e4_wave.py download --ckpt sft  --dest /ckpts/base-lora
    python $P/e4_wave.py download --ckpt grpo --dest /ckpts/grpo
    python $P/e4_wave.py init --sft /ckpts/base-lora --grpo /ckpts/grpo
    python $P/e4_wave.py s0                  # smoke on both checkpoints
    python $P/e4_wave.py s0-envs20           # optional: may num_envs be 20?
    python $P/e4_wave.py lock --num-envs 10  # freezes num_envs, MUJOCO_GL, speed
    python $P/e4_wave.py s1                  # pilot + pilot probes + gates
    python $P/e4_wave.py s2                  # 18 collections, linear probes behind
    python $P/e4_wave.py s3 --wait           # on a CPU pod: MLP, goal runs only
    python $P/e4_wave.py status

The collection and probe code are not touched: this file only builds
``run_collect.py`` / ``run_probes.py`` command lines from ``e4_plan.yaml`` and
checks what they wrote.  What it adds is the operator discipline wave 2 had to
learn the hard way (see its report): completion is the child's exit code plus
content checks, never the incrementally rewritten manifest; a run directory is
never reused; kills go to our own child's process group, never ``pkill``;
``--jobs`` is derived from the CPU quota, never ``-1``; disk headroom is a real
write, never ``df``; and a run at 2x its expected wall clock is stopped and
reported.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from efference_probe import e4_common as common  # noqa: E402
from efference_probe.config import load_config  # noqa: E402
from efference_probe.e4_ckpt_check import verify_local  # noqa: E402

GATE_BAND = (0.45, 0.55)
MAX_LOGPROB_DIFF = 1e-3
S0_HIDDEN_TAIL = (3, 3, 4096)  # smoke.yaml: layers [0, 16, 32] x 3 pools x hidden
ENVS20_MEMORY_FRACTION = 0.85
ENVS20_VRAM_FRACTION = 0.90
# Amendment 1 (pilot2 gate only): a single-cell control passes when it is
# within this many fold sds of chance, and the gate counts only with at least
# this many probe samples per class.  The strict band above stays reported.
PILOT_CONTROL_SDS = 2.0
PILOT_MIN_PER_CLASS = 40
MAX_FAILED_ATTEMPTS = 2  # S2 only: a run that fails twice halts the GPU lane
COLLECTOR_CPU_RESERVE = 4  # probe workers leave this many CPUs to env stepping


class StageError(RuntimeError):
    """A precondition for a stage is not met; nothing was started."""


# -- the box ---------------------------------------------------------------


def cgroup_cpu_quota(root: Path = Path("/sys/fs/cgroup")) -> Optional[float]:
    """CPU quota in cores, or None if unlimited / not under a cgroup."""
    v2 = root / "cpu.max"
    if v2.is_file():
        quota, period = v2.read_text().split()[:2]
        return None if quota == "max" else int(quota) / int(period)
    quota_path, period_path = (
        root / "cpu/cpu.cfs_quota_us",
        root / "cpu/cpu.cfs_period_us",
    )
    if quota_path.is_file() and period_path.is_file():
        quota = int(quota_path.read_text())
        return None if quota <= 0 else quota / int(period_path.read_text())
    return None


def usable_cpus(root: Path = Path("/sys/fs/cgroup")) -> int:
    """Cores this process may actually use.

    ``os.cpu_count()`` reports the host (112 and 192 on the two wave-2 boxes,
    against quotas of 23.8 and 16), and ``sched_getaffinity`` does not see a
    CFS quota either, so ``--jobs -1`` oversubscribed both boxes.
    """
    count = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        count = min(count, len(os.sched_getaffinity(0)))
    quota = cgroup_cpu_quota(root)
    if quota is not None:
        count = min(count, max(1, int(quota)))
    return count


def memory_limit_bytes(root: Path = Path("/sys/fs/cgroup")) -> int:
    """The binding memory limit: cgroup if set, else physical RAM."""
    total = 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) * 1024
    except OSError:
        pass
    for path in (root / "memory.max", root / "memory/memory.limit_in_bytes"):
        if path.is_file():
            raw = path.read_text().strip()
            if raw != "max" and int(raw) < (total or float("inf")):
                return int(raw)
    return total


def tree_rss_bytes(pid: int) -> int:
    """Resident memory of ``pid`` and all its descendants (env workers included).

    Not cgroup ``memory.current``: that counts page cache, and reading a 15 GB
    checkpoint would make a healthy run look close to its limit.
    """
    children: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            ppid = int(stat[stat.rfind(")") + 2 :].split()[1])
            for line in (entry / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss[int(entry.name)] = int(line.split()[1]) * 1024
            children.setdefault(ppid, []).append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    total, stack = 0, [pid]
    while stack:
        current = stack.pop()
        total += rss.get(current, 0)
        stack.extend(children.get(current, []))
    return total


def gpu_memory_mib() -> Optional[tuple[int, int]]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout.splitlines()[0]
        used, total = (int(x) for x in out.split(","))
        return used, total
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def headroom_ok(path: Path, gigabytes: float) -> tuple[bool, str]:
    """Write and delete a real file: ``df`` on a network volume reports the pool.

    Wave 2 ran into an exhausted quota while ``df`` read 485 TB free.
    """
    probe = path / f".e4_headroom_{os.getpid()}"
    block = b"\0" * (1 << 20)
    try:
        path.mkdir(parents=True, exist_ok=True)
        with open(probe, "wb") as handle:
            for _ in range(int(gigabytes * 1024)):
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        return True, f"{gigabytes:g} GB write ok"
    except OSError as error:
        return False, f"{gigabytes:g} GB write failed: {error}"
    finally:
        try:
            probe.unlink()
        except OSError:
            pass


# -- running children ------------------------------------------------------


class _Sampler:
    """Peak memory of the child's process tree and of the GPU, sampled."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.peak_rss = 0
        self.peak_vram: Optional[int] = None
        self.vram_total: Optional[int] = None
        self._ticks = 0

    def sample(self) -> None:
        self.peak_rss = max(self.peak_rss, tree_rss_bytes(self.pid))
        self._ticks += 1
        if self._ticks % 6 == 1:  # nvidia-smi is a process spawn; every ~30 s
            gpu = gpu_memory_mib()
            if gpu is not None:
                self.peak_vram = max(self.peak_vram or 0, gpu[0])
                self.vram_total = gpu[1]


def _kill_group(proc: subprocess.Popen, log) -> None:
    """Stop our own child: SIGINT (the harness finalises its rows), then harder.

    Signals go to the child's process group, which ``start_new_session`` made
    ours alone -- never a ``pkill`` pattern, which in wave 2 matched joblib
    workers and the agent's own shell.
    """
    for sig, grace in (
        (signal.SIGINT, 120),
        (signal.SIGTERM, 30),
        (signal.SIGKILL, 30),
    ):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        log(f"watchdog: sent {sig.name} to process group {proc.pid}")
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def run_child(
    cmd: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
    kill_after_s: float,
    log,
    poll_s: float = 5.0,
) -> dict[str, Any]:
    """Run ``cmd``, tee its output, sample memory, and enforce the watchdog.

    The live log goes to local disk and is copied to ``log_path`` in one write
    at the end: wave 2's long ``tee -a`` appends on a network volume came back
    with NUL holes and lost two tracebacks.
    """
    local = Path(tempfile.mkdtemp(prefix="e4_")) / log_path.name
    start = time.time()
    killed = False
    with open(local, "wb") as sink:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        def pump() -> None:
            out = getattr(sys.stdout, "buffer", None)
            for line in iter(proc.stdout.readline, b""):
                sink.write(line)
                sink.flush()
                if out is not None:
                    out.write(line)
                else:
                    sys.stdout.write(line.decode(errors="replace"))
                sys.stdout.flush()

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        sampler = _Sampler(proc.pid)
        while proc.poll() is None:
            sampler.sample()
            if not killed and time.time() - start > kill_after_s:
                killed = True
                log(
                    f"watchdog: `{Path(cmd[1]).name}` passed {kill_after_s / 3600:.2f} h "
                    f"(pid {proc.pid}); stopping it"
                )
                _kill_group(proc, log)
            time.sleep(poll_s)
        reader.join(timeout=60)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(local.read_bytes())
    return {
        "rc": proc.returncode,
        "elapsed_s": round(time.time() - start, 1),
        "watchdog_killed": killed,
        "peak_tree_rss_gb": round(sampler.peak_rss / 2**30, 2),
        "peak_vram_mib": sampler.peak_vram,
        "vram_total_mib": sampler.vram_total,
        "log": str(log_path),
    }


# -- the wave --------------------------------------------------------------


class Wave:
    """Paths and state for one E4 wave under ``data_root``."""

    def __init__(self, data_root: Path, logbook: Path, plan: dict[str, Any]) -> None:
        self.data_root = data_root
        self.logbook = logbook
        self.plan = plan
        self.state_path = data_root / "e4_state.json"
        self.halt_path = data_root / "E4_HALT"
        self._log_lock = threading.Lock()

    # bookkeeping

    def log(self, message: str) -> None:
        """Logbook line in the harness's own format, plus stdout."""
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{stamp}] {message}", flush=True)
        with self._log_lock:
            self.logbook.parent.mkdir(parents=True, exist_ok=True)
            with open(self.logbook, "a", encoding="utf-8") as handle:
                handle.write(f"- `{stamp}` **E4 (driver)** {message}\n")

    @property
    def state(self) -> dict[str, Any]:
        return common.read_json(self.state_path) or {}

    def update_state(self, **changes: Any) -> None:
        state = self.state
        state.update(changes)
        common.write_json_atomic(self.state_path, state)

    def gate_path(self, stage: str, ckpt: str) -> Path:
        return self.data_root / "e4_gates" / f"{stage}_{ckpt}.json"

    def gate(self, stage: str, ckpt: str) -> Optional[dict[str, Any]]:
        return common.read_json(self.gate_path(stage, ckpt))

    def record(
        self, kind: str, run_id: str, result: dict[str, Any], outcome: str
    ) -> None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        end = time.time()
        common.write_json_atomic(
            common.ledger_dir(self.data_root) / f"{stamp}_{kind}_{run_id}.json",
            {
                "start_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(end - result["elapsed_s"])
                ),
                "end_utc": common.utc_now(),
                "kind": kind,
                "run_id": run_id,
                "rc": result["rc"],
                "elapsed_s": result["elapsed_s"],
                "outcome": outcome,
                "watchdog_killed": result["watchdog_killed"],
                "peak_tree_rss_gb": result["peak_tree_rss_gb"],
                "peak_vram_mib": result["peak_vram_mib"],
                "host": os.uname().nodename,
            },
        )

    def halt(self, reason: str) -> None:
        self.halt_path.write_text(f"{common.utc_now()} {reason}\n", encoding="utf-8")
        self.log(
            f"**HALT** {reason} -- the GPU lane is stopped until `e4_wave.py resume`"
        )

    def require_not_halted(self) -> None:
        if self.halt_path.exists():
            raise StageError(
                f"halted: {self.halt_path.read_text().strip()} "
                "(review, then `e4_wave.py resume --reason ...`)"
            )

    # checkpoints and lock

    def ckpt_path(self, ckpt: str) -> str:
        path = self.state.get("checkpoints", {}).get(ckpt, {}).get("path")
        if not path:
            raise StageError("checkpoints not registered; run `e4_wave.py init` first")
        return path

    @property
    def lock(self) -> dict[str, Any]:
        lock = self.state.get("lock")
        if not lock:
            raise StageError(
                "wave constants not locked; run `e4_wave.py lock` after S0"
            )
        return lock


def sim_env(mujoco_gl: str) -> dict[str, str]:
    """The variables evaluations/run_eval.sh exports, plus the rendering backend."""
    env = dict(os.environ)
    env.update(
        MUJOCO_GL=mujoco_gl,
        PYOPENGL_PLATFORM=mujoco_gl,
        ROBOT_PLATFORM="LIBERO",
        LIBERO_TYPE="standard",
        TOKENIZERS_PARALLELISM="false",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(common.REPO_ROOT), env.get("PYTHONPATH", "")) if p
    )
    # Wave 2: a template-exported HF_HUB_ENABLE_HF_TRANSFER broke HF calls
    # once uv pruned hf_transfer from the venv.
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    return env


def probe_env(jobs: int) -> dict[str, str]:
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = str(jobs)
    return env


def resolve_jobs(requested: Optional[int], reserve: int = 0) -> int:
    if requested is not None:
        if requested < 1:
            raise StageError("--jobs must be a positive worker count; -1 is never used")
        return requested
    return max(1, usable_cpus() - reserve)


# -- collection ------------------------------------------------------------


def collection_checks(run_dir: Path, rc: int) -> dict[str, Any]:
    """Did the child produce a complete, internally consistent run?"""
    manifest = common.read_json(run_dir / "manifest.json") or {}
    checks: dict[str, Any] = {"rc_zero": rc == 0}
    checks["stop_reason"] = manifest.get("stop_reason")
    checks["stop_reason_completed"] = manifest.get("stop_reason") == "completed"
    checks["verification_passed"] = bool(manifest.get("verification", {}).get("passed"))
    parquet = run_dir / "calls.parquet"
    rows = len(pd.read_parquet(parquet)) if parquet.is_file() else -1
    checks["n_calls_manifest"] = manifest.get("n_calls")
    checks["n_calls_parquet"] = rows
    checks["rows_match_manifest"] = rows >= 0 and rows == manifest.get("n_calls")
    hidden = common.hidden_integrity(run_dir) if (run_dir / "hidden").is_dir() else {}
    checks["hidden"] = hidden
    checks["hidden_matches_rows"] = (
        bool(hidden)
        and not hidden["bad_archives"]
        and hidden["n_hidden_calls"] == rows
        and hidden["n_archives"] == manifest.get("n_episodes")
    )
    checks["passed"] = all(
        checks[k]
        for k in (
            "rc_zero",
            "stop_reason_completed",
            "verification_passed",
            "rows_match_manifest",
            "hidden_matches_rows",
        )
    )
    return checks


def collect_overrides(
    plan: dict[str, Any],
    run: common.PlannedRun,
    run_id: str,
    ckpt_path: str,
    data_root: Path,
    num_envs: Optional[int],
    budget_s: float,
    init_states: Optional[int] = None,
) -> list[str]:
    """``--set`` overrides on top of the unchanged wave-2 config.

    This list *is* the difference between an E4 run and its wave-2 sibling;
    ``tests/test_e4.py`` resolves every planned run through ``load_config``
    and fails if any other field moved.  ``init_states`` is Amendment 1's
    sample-matched episode count, passed only for the sized checkpoint's S2
    arms.
    """
    overrides = [
        f"model.model_path={ckpt_path}",
        f"model.unnorm_key={plan['unnorm_key']}",
        f"run_id={run_id}",
        f"out_root={data_root}",
        f"budget.max_gpu_hours={budget_s / 3600:.4f}",
        f"notes=E4 {run.ckpt} {run.key} ({plan['checkpoints'][run.ckpt]['label']})",
    ]
    if num_envs is not None:
        overrides.append(f"env.num_envs={num_envs}")
    if run.seed is not None:
        overrides.append(f"env.seed={run.seed}")
    if init_states is not None:
        overrides.append(f"env.init_states_per_task={init_states}")
    if run.decoding:
        overrides.extend(plan["decoding_overrides"])
    overrides.extend(run.overrides)  # entry-specific last, e.g. smoke20's num_envs
    return overrides


def base_states(run: common.PlannedRun) -> int:
    """``init_states_per_task`` of the run's own wave-2 config."""
    return int(
        load_config(str(common.CONFIG_DIR / run.config)).env.init_states_per_task
    )


def sized_states(wave: Wave, run: common.PlannedRun) -> Optional[int]:
    """Amendment 1's episode count for this run, or None if it is not sized.

    Only the sized checkpoint's S2 arms are; their collection refuses to start
    until ``e4_wave.py size`` has frozen the numbers.
    """
    sizing_plan = wave.plan.get("sizing")
    if run.stage != "s2" or not sizing_plan or run.ckpt != sizing_plan["sized_ckpt"]:
        return None
    sizing = wave.state.get("sizing")
    if not sizing:
        raise StageError(
            f"`{run.run_id}` is sized by Amendment 1; run `e4_wave.py size` first"
        )
    return int(sizing["states_per_task"][run.key])


def collect(
    wave: Wave, run: common.PlannedRun, mujoco_gl: Optional[str] = None
) -> dict:
    """Collect one planned run, or return the attempt that already completed."""
    data_root = wave.data_root
    done = common.completed_attempt(data_root, run.run_id)
    if done:
        print(f"{run.run_id}: already collected as {done}; skipping", flush=True)
        return common.job_status(data_root, done, "collect")
    attempts = common.attempt_ids(data_root, run.run_id)
    if run.stage == "s2" and len(attempts) >= MAX_FAILED_ATTEMPTS:
        wave.halt(f"{run.run_id} has {len(attempts)} incomplete attempts {attempts}")
        raise StageError(f"{run.run_id}: two-strike rule")
    run_id = common.next_attempt_id(data_root, run.run_id)
    if attempts:
        wave.log(
            f"`{run.run_id}`: earlier attempt(s) {attempts} never completed; "
            f"collecting as `{run_id}` (interrupted collection = new id; "
            "nothing overwritten or deleted)"
        )

    plan_budget = wave.plan["budget"]
    if run.stage in ("s1", "s1b", "s2"):
        lock = wave.lock
        num_envs, speed = int(lock["num_envs"]), float(lock["speed_factor"])
        mujoco_gl = lock["mujoco_gl"]
    else:
        num_envs, speed = None, 1.0
        mujoco_gl = mujoco_gl or os.environ.get("MUJOCO_GL", "egl")
    expected_s = run.expected_s * speed
    init_states = sized_states(wave, run)
    if init_states is not None:
        # More episodes than the wave-2 config: scale the expectation with the
        # episode count.  An upper bound, since the sized checkpoint is the one
        # whose episodes end early.
        expected_s *= init_states / base_states(run)
    remaining_s = (
        plan_budget["wave_gpu_hours"] * 3600 - common.gpu_hours_used(data_root) * 3600
    )
    if expected_s > remaining_s:
        message = (
            f"`{run_id}` expects {expected_s / 3600:.2f} h but only "
            f"{remaining_s / 3600:.2f} h of the {plan_budget['wave_gpu_hours']} h "
            "wave budget remain; cut from the tail"
        )
        if run.stage == "s2":
            wave.halt(message)
        raise StageError(message)
    budget_s = min(plan_budget["watchdog_factor"] * expected_s, remaining_s)
    ok, why = headroom_ok(data_root, wave.state.get("headroom_gb", 1.0))
    if not ok:
        if run.stage == "s2":
            wave.halt(f"disk headroom: {why}")
        raise StageError(f"disk headroom: {why}")

    overrides = collect_overrides(
        wave.plan,
        run,
        run_id,
        wave.ckpt_path(run.ckpt),
        data_root,
        num_envs,
        budget_s,
        init_states=init_states,
    )
    cmd = [
        sys.executable,
        # Test seam: tests/fake_collect.py stands in for the GPU collector so
        # the whole stage flow runs offline.  Never set on a real box.
        os.environ.get("E4_COLLECT_SCRIPT", str(common.PROBE_DIR / "run_collect.py")),
        "--config",
        str(common.CONFIG_DIR / run.config),
        "--logbook",
        str(wave.logbook),
        "--set",
        *overrides,
    ]
    wave.log(
        f"collect `{run_id}` ({run.stage}, {run.ckpt}, {run.config}); expected "
        f"{expected_s / 60:.0f} min, stop at {budget_s / 60:.0f} min; "
        f"MUJOCO_GL={mujoco_gl}; GPU used so far "
        f"{common.gpu_hours_used(data_root):.2f}/{plan_budget['wave_gpu_hours']} h"
    )
    result = run_child(
        cmd,
        env=sim_env(mujoco_gl),
        log_path=data_root / "e4_logs" / f"{run_id}.collect.log",
        kill_after_s=budget_s + plan_budget["hang_grace_s"],
        log=wave.log,
    )
    run_dir = data_root / run_id
    checks = collection_checks(run_dir, result["rc"])
    status = {
        "run_id": run_id,
        "planned_id": run.run_id,
        "stage": run.stage,
        "ckpt": run.ckpt,
        "key": run.key,
        "suite": run.suite,
        "transform": run.transform,
        "mujoco_gl": mujoco_gl,
        "num_envs": num_envs,
        "expected_s": expected_s,
        "budget_s": budget_s,
        "finished_utc": common.utc_now(),
        "ok": bool(checks["passed"]),
        **result,
        "checks": checks,
    }
    common.write_json_atomic(common.status_path(data_root, run_id, "collect"), status)
    outcome = "ok" if status["ok"] else f"failed ({checks['stop_reason']})"
    wave.record("collect", run_id, result, outcome)
    wave.log(
        f"collect `{run_id}` {outcome}: rc={result['rc']} "
        f"{result['elapsed_s'] / 60:.1f} min (expected {expected_s / 60:.0f}), "
        f"calls={checks['n_calls_parquet']}, peak RSS {result['peak_tree_rss_gb']} GB, "
        f"peak VRAM {result['peak_vram_mib']} MiB"
    )
    if run.stage == "s2" and not status["ok"]:
        if (
            checks["stop_reason"] == "time_budget_exhausted"
            or result["watchdog_killed"]
        ):
            wave.halt(
                f"`{run_id}` reached {plan_budget['watchdog_factor']}x its expected "
                f"wall clock ({result['elapsed_s'] / 60:.0f} min vs "
                f"{expected_s / 60:.0f} min expected); stopped and reported"
            )
        else:
            wave.halt(f"`{run_id}` failed its completion checks: {outcome}")
    return status


def write_competence(wave: Wave, run_id: str) -> pd.DataFrame:
    frame = common.competence_report(wave.data_root / run_id)
    out = common.derived_dir(wave.data_root, run_id)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "competence.csv", index=False)
    flagged = frame.loc[frame["flagged"], "task_id"].tolist() if not frame.empty else []
    if flagged:
        wave.log(
            f"`{run_id}`: low-competence task flags {flagged} (kept in every "
            "analysis; see e4_derived/<run>/competence.csv)"
        )
    return frame


# -- probing -----------------------------------------------------------------


def probe(wave: Wave, run_id: str, family: str, jobs: int) -> dict[str, Any]:
    """One ``run_probes.py`` round, with the wave-2 gates around it."""
    data_root = wave.data_root
    kind = f"probe-{family}"
    existing = common.job_status(data_root, run_id, kind)
    if existing and existing.get("ok"):
        print(f"{run_id}: {kind} already complete; skipping", flush=True)
        return existing
    collected = common.job_status(data_root, run_id, "collect")
    if not (collected and collected.get("ok")):
        raise StageError(f"{run_id}: collection is not complete; refusing to probe")
    if family == "mlp":
        linear = common.job_status(data_root, run_id, "probe-linear")
        if not (linear and linear.get("ok")):
            raise StageError(f"{run_id}: MLP waits for a passing linear round (gate 2)")

    run_dir = data_root / run_id
    out = common.analysis_dir(run_dir, family)
    if out.exists():
        moved = out.with_name(
            f"{out.name}.partial_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        )
        out.rename(moved)
        wave.log(
            f"`{run_id}`: half-written `{out.name}` renamed to `{moved.name}` (not deleted)"
        )

    spec = wave.plan["probes"][family]
    cmd = [
        sys.executable,
        str(common.PROBE_DIR / "run_probes.py"),
        "--run",
        str(run_dir),
        "--jobs",
        str(jobs),
        *[str(a) for a in spec["args"]],
    ]
    kill_after = wave.plan["budget"]["watchdog_factor"] * spec["expected_s"]
    wave.log(
        f"probe `{run_id}` {family} (--jobs {jobs}); stop at {kill_after / 3600:.1f} h"
    )
    result = run_child(
        cmd,
        env=probe_env(jobs),
        log_path=data_root / "e4_logs" / f"{run_id}.{kind}.log",
        kill_after_s=kill_after,
        log=wave.log,
    )
    plumbing = common.read_json(out / "plumbing.json") or {}
    ok = (
        result["rc"] == 0
        and bool(plumbing.get("passed"))
        and (out / "probe_results.csv").is_file()
        and (out / "summary.md").is_file()
    )
    status = {
        "run_id": run_id,
        "family": family,
        "jobs": jobs,
        "ok": ok,
        "plumbing_passed": bool(plumbing.get("passed")),
        "finished_utc": common.utc_now(),
        **result,
    }
    common.write_json_atomic(common.status_path(data_root, run_id, kind), status)
    wave.record(kind, run_id, result, "ok" if ok else "failed")
    wave.log(
        f"probe `{run_id}` {family} {'ok' if ok else 'FAILED'}: rc={result['rc']} "
        f"{result['elapsed_s'] / 60:.1f} min, plumbing.passed={plumbing.get('passed')}"
    )
    return status


# -- gates -------------------------------------------------------------------


def gate_s0(run_dir: Path, collected: dict[str, Any]) -> dict[str, Any]:
    manifest = common.read_json(run_dir / "manifest.json") or {}
    verification = manifest.get("verification", {})
    checks: dict[str, Any] = {"collection_ok": bool(collected.get("ok"))}
    checks["indexing_check_passed"] = bool(verification.get("passed"))
    diff = verification.get("max_abs_logprob_diff")
    checks["max_abs_logprob_diff_below_1e-3"] = (
        diff is not None and diff < MAX_LOGPROB_DIFF
    )
    calls = (
        pd.read_parquet(run_dir / "calls.parquet")
        if (run_dir / "calls.parquet").is_file()
        else None
    )
    checks["no_null_cells"] = calls is not None and int(calls.isna().sum().sum()) == 0
    hidden = common.hidden_integrity(run_dir, expect_tail=S0_HIDDEN_TAIL)
    checks["hidden_shape_n_3_3_4096"] = (
        bool(hidden.get("shape_ok")) and not hidden["bad_archives"]
    )
    checks["hidden_calls_match_rows"] = calls is not None and hidden[
        "n_hidden_calls"
    ] == len(calls)
    transform = _run_transform(run_dir)
    checks["labels_follow_schedule"] = (
        calls is not None
        and "HIJACK" in set(calls["label"])
        and set(calls["transform"]) <= {"", transform}
    )
    return {
        "stage": "s0",
        "run_id": run_dir.name,
        "passed": all(v for v in checks.values()),
        "checks": checks,
        "verification": verification,
        "hidden": hidden,
        "mujoco_gl": collected.get("mujoco_gl"),
        "evaluated_utc": common.utc_now(),
    }


def _run_transform(run_dir: Path) -> str:
    path = run_dir / "run_config.yaml"
    if not path.is_file():
        return ""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)["config"]["hijack"]["transform"]


def gate_envs20(collected: dict[str, Any]) -> dict[str, Any]:
    limit = memory_limit_bytes()
    rss = collected.get("peak_tree_rss_gb", 0) * 2**30
    vram, vram_total = collected.get("peak_vram_mib"), collected.get("vram_total_mib")
    checks = {
        "collection_ok": bool(collected.get("ok")),
        "memory_below_85pct": bool(limit) and rss < ENVS20_MEMORY_FRACTION * limit,
        # Fail closed: no measurement is no licence to double the env count.
        "vram_below_90pct": vram is not None
        and bool(vram_total)
        and vram < ENVS20_VRAM_FRACTION * vram_total,
    }
    return {
        "stage": "s0_envs20",
        "run_id": collected.get("run_id"),
        "passed": all(checks.values()),
        "checks": checks,
        "peak_tree_rss_gb": collected.get("peak_tree_rss_gb"),
        "memory_limit_gb": round(limit / 2**30, 1),
        "peak_vram_mib": vram,
        "vram_total_mib": vram_total,
        "evaluated_utc": common.utc_now(),
    }


def step_index_overlap(path: Path) -> dict[str, Any]:
    """Each class's median call index inside the other's 5-95% range."""
    frame = pd.read_csv(path / "step_index_report.csv", index_col=0)
    frame.index = frame.index.astype(int)
    if not {0, 1} <= set(frame.index):
        return {"overlap": False, "note": "a class is missing"}
    neg, pos = frame.loc[0], frame.loc[1]
    overlap = bool(
        pos["p05"] <= neg["median"] <= pos["p95"]
        and neg["p05"] <= pos["median"] <= neg["p95"]
    )
    return {"overlap": overlap, "table": frame.round(2).to_dict(orient="index")}


def gate_s1(
    run_dir: Path,
    collected: dict,
    probed: dict,
    competence: pd.DataFrame,
    rule: str = "strict",
    stage: str = "s1",
) -> dict:
    """The README's S1 gate.

    ``rule="strict"`` is the gate as registered on 2026-10-26 03:10 (C_cmd and
    C_phase inside [0.45, 0.55]).  ``rule="amended"`` is Amendment 1, used for
    pilot2: those two controls pass within ``PILOT_CONTROL_SDS`` fold sds of
    0.5, and the gate needs ``PILOT_MIN_PER_CLASS`` samples per class to count.
    """
    out = common.analysis_dir(run_dir, "pilot")
    checks: dict[str, Any] = {
        "collection_ok": bool(collected.get("ok")),
        "probe_round_ok": bool(probed.get("ok")),
    }
    report: dict[str, Any] = {}
    if probed.get("ok"):
        plumbing = common.read_json(out / "plumbing.json") or {}
        checks["plumbing_passed_zero_mismatches"] = bool(
            plumbing.get("passed")
        ) and all(
            int(plumbing.get(k, 1)) == 0
            for k in (
                "n_action_mismatches",
                "n_missing_donors",
                "n_unknown_labels",
                "n_state_discontinuities",
            )
        )
        frame = common.probe_frame(out)
        p0 = common.best_cell(frame, "P0")
        # P0's gate is on the mean over its 27 cells: the *maximum* of a null
        # over 27 tries sits ~+0.06-0.12 above chance at pilot n (paper T1),
        # and is reported as the pilot's selection floor instead.
        checks["P0_mean_over_cells_in_band"] = (
            p0 is not None and GATE_BAND[0] <= p0["mean_over_cells"] <= GATE_BAND[1]
        )
        extra = common.summary_extra(out)
        e3 = extra.get("e3", {})
        for name in ("C_cmd", "C_phase"):
            cell = common.best_cell(frame, name)
            in_band = cell is not None and GATE_BAND[0] <= cell["bacc"] <= GATE_BAND[1]
            if rule == "strict":
                checks[f"{name}_in_band"] = in_band
            else:
                checks[f"{name}_within_2sd_of_chance"] = (
                    cell is not None
                    and abs(cell["bacc"] - 0.5) <= PILOT_CONTROL_SDS * cell["sd"]
                )
                report[f"{name}_in_strict_band"] = in_band
            report[name] = cell
        n_pos, n_neg = int(e3.get("n_positive", 0)), int(e3.get("n_negative", 0))
        if rule != "strict":
            checks["min_samples_per_class"] = min(n_pos, n_neg) >= PILOT_MIN_PER_CLASS
        manifest = common.read_json(run_dir / "manifest.json") or {}
        episodes = [
            e for e in manifest.get("episodes", []) if not e.get("censored", False)
        ]
        overlap = step_index_overlap(out)
        checks["step_index_overlap"] = overlap["overlap"]
        report.update(
            n_positive=n_pos,
            n_negative=n_neg,
            n_episodes=len(episodes),
            # Amendment 1's sizing input: kept positives per finished episode.
            positives_per_episode=(n_pos / len(episodes)) if episodes else None,
        )
        report.update(
            P0=p0,
            P1=common.best_cell(frame, "P1"),
            P2=common.best_cell(frame, "P2"),
            P2r=common.best_cell(frame, "P2r"),
            C_dstates=common.best_cell(frame, "C_dstates"),
            step_index=overlap,
            plumbing=plumbing,
            e3_success=e3.get("success_rate"),
        )
    report["competence"] = (
        competence.to_dict(orient="records") if not competence.empty else []
    )
    report["flagged_tasks"] = (
        competence.loc[competence["flagged"], "task_id"].tolist()
        if not competence.empty
        else []
    )
    return {
        "stage": stage,
        "rule": rule,
        "run_id": run_dir.name,
        "passed": all(bool(v) for v in checks.values()),
        "checks": checks,
        "report": report,
        "evaluated_utc": common.utc_now(),
    }


def _save_gate(wave: Wave, stage: str, ckpt: str, gate: dict[str, Any]) -> None:
    common.write_json_atomic(wave.gate_path(stage, ckpt), gate)
    failed = [k for k, v in gate["checks"].items() if not v]
    wave.log(
        f"gate {stage} {ckpt} (`{gate['run_id']}`): "
        + ("**PASSED**" if gate["passed"] else f"**FAILED** {failed}")
    )


# -- commands ------------------------------------------------------------------


def cmd_download(wave: Wave, args) -> int:
    from huggingface_hub import snapshot_download

    spec = wave.plan["checkpoints"][args.ckpt]
    snapshot_download(
        repo_id=spec["hf_repo"],
        revision=spec["revision"],
        local_dir=args.dest,
        ignore_patterns=spec.get("download_exclude"),
    )
    report = verify_local(args.dest, spec, wave.plan["unnorm_key"])
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def cmd_init(wave: Wave, args) -> int:
    checkpoints = {}
    for ckpt, path in (("sft", args.sft), ("grpo", args.grpo)):
        spec = wave.plan["checkpoints"][ckpt]
        report = verify_local(
            path, spec, wave.plan["unnorm_key"], hash_shards=not args.no_hash
        )
        if not report["passed"]:
            print(json.dumps(report, indent=2))
            raise StageError(f"{ckpt} checkpoint at {path} does not match the plan")
        checkpoints[ckpt] = {
            "path": str(Path(path).resolve()),
            "hf_repo": spec["hf_repo"],
            "revision": spec["revision"],
            "sha256_verified": not args.no_hash,
        }
    wave.update_state(
        checkpoints=checkpoints,
        headroom_gb=args.headroom_gb,
        created_utc=common.utc_now(),
    )
    wave.log(
        "checkpoints registered: "
        + ", ".join(
            f"{k}=`{v['hf_repo']}@{v['revision'][:12]}` at `{v['path']}`"
            f" (sha256 {'verified' if v['sha256_verified'] else 'NOT checked'})"
            for k, v in checkpoints.items()
        )
    )
    return 0


def cmd_s0(wave: Wave, args, stage: str = "s0") -> int:
    passed = True
    for run in common.planned_runs(wave.plan, stage):
        if args.ckpt and run.ckpt not in args.ckpt:
            continue
        collected = collect(wave, run)
        run_dir = wave.data_root / collected["run_id"]
        gate = gate_s0(run_dir, collected) if stage == "s0" else gate_envs20(collected)
        _save_gate(wave, stage, run.ckpt, gate)
        print(json.dumps(gate, indent=2, default=str))
        passed &= gate["passed"]
    return 0 if passed else 1


def cmd_lock(wave: Wave, args) -> int:
    gates = {ckpt: wave.gate("s0", ckpt) for ckpt in common.CKPTS}
    if not all(g and g["passed"] for g in gates.values()):
        raise StageError("both S0 gates must pass before the wave constants are locked")
    started = [
        p.name
        for stage in ("s1", "s1b", "s2")
        for run in common.planned_runs(wave.plan, stage)
        for p in wave.data_root.glob(f"{run.run_id}*")
    ]
    if wave.state.get("lock") and started:
        raise StageError(
            f"constants are already locked and runs exist ({started[:3]}...)"
        )
    if args.num_envs not in (10, 20):
        raise StageError(
            "num_envs is 10 (wave-2 default) or 20 (only if S0-envs20 passed)"
        )
    if args.num_envs == 20:
        envs20 = {ckpt: wave.gate("s0_envs20", ckpt) for ckpt in common.CKPTS}
        if not all(g and g["passed"] for g in envs20.values()):
            raise StageError(
                "num_envs=20 needs a passing `s0-envs20` gate on both checkpoints"
            )
    backends = {g["mujoco_gl"] for g in gates.values()}
    if len(backends) != 1:
        raise StageError(f"S0 ran under different MUJOCO_GL backends: {backends}")
    lock = {
        "num_envs": args.num_envs,
        "mujoco_gl": backends.pop(),
        "speed_factor": args.speed_factor,
        "gpu": (gpu_memory_mib() or (None, None))[1],
        "locked_utc": common.utc_now(),
    }
    wave.update_state(lock=lock)
    wave.log(
        f"wave constants locked for every S1/S2 arm: num_envs={lock['num_envs']}, "
        f"MUJOCO_GL={lock['mujoco_gl']}, speed_factor={lock['speed_factor']}"
    )
    return 0


def cmd_s1(wave: Wave, args, stage: str = "s1") -> int:
    """S1 (the registered pilot) or S1b (Amendment 1's pilot2) and its gate.

    Each stage writes its own gate files, so the S1 record stays as it was.
    """
    wave.lock  # noqa: B018 - raises if not locked
    rule = "strict" if stage == "s1" else "amended"
    passed = True
    for run in common.planned_runs(wave.plan, stage):
        if args.ckpt and run.ckpt not in args.ckpt:
            continue
        collected = collect(wave, run)
        run_id = collected["run_id"]
        probed: dict[str, Any] = {}
        competence = pd.DataFrame()
        if collected["ok"]:
            probed = probe(wave, run_id, "pilot", resolve_jobs(args.jobs))
            competence = write_competence(wave, run_id)
        gate = gate_s1(
            wave.data_root / run_id, collected, probed, competence, rule, stage
        )
        _save_gate(wave, stage, run.ckpt, gate)
        print(json.dumps(gate, indent=2, default=str))
        passed &= gate["passed"]
    return 0 if passed else 1


def _missing_markers() -> list[str]:
    """Registration entries S2 depends on that the canonical logbook lacks."""
    logbook = common.PROBE_DIR / "logbook.md"
    text = logbook.read_text(encoding="utf-8") if logbook.is_file() else ""
    return [
        m for m in (common.PREREG_MARKER, *common.AMENDMENT_MARKERS) if m not in text
    ]


def require_s2_ready(wave: Wave) -> None:
    """S2 starts only after the S1b gates, the sizing, and their registration."""
    for ckpt in common.CKPTS:
        gate = wave.gate("s1b", ckpt)
        if not (gate and gate["passed"]):
            raise StageError(
                f"S1b (pilot2) gate for {ckpt} has not passed; S2 does not start"
            )
    if wave.plan.get("sizing") and not wave.state.get("sizing"):
        raise StageError("GRPO arms are not sized yet; run `e4_wave.py size`")
    missing = _missing_markers()
    if missing:
        raise StageError(
            f"logbook.md lacks {missing}: every rule S2 depends on must be "
            "registered before any S2 data exists"
        )


def cmd_size(wave: Wave, args) -> int:
    """Freeze Amendment 1's GRPO episode counts from the pilot2 yields.

    k = min(cap, max(k_base, ceil(k_base * r))) with r the reference
    checkpoint's kept positives per episode over the sized one's.  Refused
    once any S2 run exists: the sizing is fixed before S2 data, like the lock.
    """
    import math

    spec = wave.plan["sizing"]
    gates = {ckpt: wave.gate(spec["yield_stage"], ckpt) for ckpt in common.CKPTS}
    if not all(g and g["passed"] for g in gates.values()):
        raise StageError("both S1b gates must pass before sizing")
    started = [
        p.name
        for run in common.planned_runs(wave.plan, "s2")
        for p in wave.data_root.glob(f"{run.run_id}*")
    ]
    if started:
        raise StageError(f"S2 runs exist ({started[:3]}...); the sizing is frozen")
    yields = {
        ckpt: gates[ckpt]["report"].get("positives_per_episode")
        for ckpt in common.CKPTS
    }
    sized, reference = spec["sized_ckpt"], spec["reference_ckpt"]
    if not yields[sized] or yields[reference] is None:
        raise StageError(f"no usable pilot2 yields: {yields}")
    ratio = yields[reference] / yields[sized]
    cap = int(spec["max_states_per_task"])
    states = {}
    for run in common.planned_runs(wave.plan, "s2"):
        if run.ckpt == sized:
            k_base = base_states(run)
            states[run.key] = min(cap, max(k_base, math.ceil(k_base * ratio)))
    sizing = {
        "ratio": ratio,
        "positives_per_episode": yields,
        "yield_runs": {c: g["run_id"] for c, g in gates.items()},
        "states_per_task": states,
        "cap": cap,
        "sized_utc": common.utc_now(),
    }
    wave.update_state(sizing=sizing)
    wave.log(
        f"Amendment 1 sizing frozen: positives/episode {reference}="
        f"{yields[reference]:.3f}, {sized}={yields[sized]:.3f}, r={ratio:.2f}; "
        f"{sized} S2 init_states_per_task {states} (cap {cap})"
    )
    print(json.dumps(sizing, indent=2))
    return 0


def cmd_s2(wave: Wave, args) -> int:
    wave.require_not_halted()
    require_s2_ready(wave)
    probe_jobs = resolve_jobs(args.probe_jobs, reserve=COLLECTOR_CPU_RESERVE)
    background: Optional[threading.Thread] = None
    for run in common.planned_runs(wave.plan, "s2"):
        if (args.ckpt and run.ckpt not in args.ckpt) or (
            args.only and run.key not in args.only
        ):
            continue
        if wave.halt_path.exists():
            break
        collected = collect(wave, run)
        if not collected["ok"]:
            break
        write_competence(wave, collected["run_id"])
        if args.probes == "none":
            continue
        if background is not None:
            background.join()  # one probe round at a time next to the collector
        job = threading.Thread(
            target=_probe_quietly,
            args=(wave, collected["run_id"], "linear", probe_jobs),
        )
        if args.probes == "inline":
            job.run()
        else:
            job.start()
            background = job
    if background is not None:
        background.join()
    cmd_status(wave, args)
    # A stopped lane must not look like success to whatever called us.
    return 1 if wave.halt_path.exists() else 0


def _probe_quietly(wave: Wave, run_id: str, family: str, jobs: int) -> None:
    """A probe failure is recorded and logged; it must not kill the GPU lane."""
    try:
        probe(wave, run_id, family, jobs)
    except Exception as error:  # noqa: BLE001
        wave.log(
            f"probe `{run_id}` {family} could not run: {type(error).__name__}: {error}"
        )


def cmd_probe(wave: Wave, args) -> int:
    status = probe(wave, args.run_id, args.family, resolve_jobs(args.jobs))
    return 0 if status.get("ok") else 1


def cmd_collect(wave: Wave, args) -> int:
    wave.require_not_halted()
    run = common.find_planned(wave.plan, args.ckpt, args.key)
    if run.stage == "s2":
        require_s2_ready(wave)
    status = collect(wave, run)
    if status["ok"] and run.stage == "s2":
        write_competence(wave, status["run_id"])
    return 0 if status["ok"] else 1


def cmd_s3(wave: Wave, args) -> int:
    suites = set(wave.plan["probes"]["mlp"]["suites"])
    runs = [r for r in common.planned_runs(wave.plan, "s2") if r.suite in suites]
    jobs = resolve_jobs(args.jobs)
    wave.log(f"S3 MLP lane up: {len(runs)} goal-suite runs, --jobs {jobs}")
    while True:
        waiting = []
        for run in runs:
            run_id = common.completed_attempt(wave.data_root, run.run_id)
            linear = run_id and common.job_status(
                wave.data_root, run_id, "probe-linear"
            )
            if not (linear and linear.get("ok")):
                waiting.append(run.run_id)
                continue
            mlp = common.job_status(wave.data_root, run_id, "probe-mlp")
            if mlp and (mlp.get("ok") or not args.retry_failed):
                continue
            _probe_quietly(wave, run_id, "mlp", jobs)
        if not args.wait or not waiting:
            break
        # A halt is written only after a collection has ended, so once it is
        # there nothing else is on its way.
        if wave.halt_path.exists():
            wave.log(
                f"S3: GPU lane halted; {len(waiting)} runs will not arrive; stopping"
            )
            break
        time.sleep(args.poll_s)
    return cmd_status(wave, args)


def cmd_competence(wave: Wave, args) -> int:
    frame = write_competence(wave, args.run_id)
    print(frame.to_string(index=False))
    return 0


def cmd_resume(wave: Wave, args) -> int:
    if not wave.halt_path.exists():
        print("not halted")
        return 0
    reason = wave.halt_path.read_text().strip()
    cleared = wave.halt_path.with_name(
        f"E4_HALT.cleared_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )
    wave.halt_path.rename(cleared)
    wave.log(
        f"HALT cleared by operator: was `{reason}`; reason for resuming: {args.reason}"
    )
    return 0


def cmd_todo(wave: Wave, args) -> int:
    """What may run next, as JSON -- the whole scheduling brain for modal_e4.py.

    A probe round with any marker (passed *or* failed) is not offered again: a
    failure is for a human to look at, not for a loop to retry.
    """
    mlp_suites = set(wave.plan["probes"]["mlp"]["suites"])
    todo: dict[str, Any] = {
        "collect": [],
        "linear": [],
        "mlp": [],
        "halted": wave.halt_path.exists(),
    }
    for run in common.planned_runs(wave.plan, "s2"):
        run_id = common.completed_attempt(wave.data_root, run.run_id)
        if run_id is None:
            todo["collect"].append([run.ckpt, run.key])
            continue
        linear = common.job_status(wave.data_root, run_id, "probe-linear")
        if linear is None:
            todo["linear"].append(run_id)
        elif (
            run.suite in mlp_suites
            and linear.get("ok")
            and common.job_status(wave.data_root, run_id, "probe-mlp") is None
        ):
            todo["mlp"].append(run_id)
    print(json.dumps(todo))
    return 0


def cmd_merge_logbook(wave: Wave, args) -> int:
    """Append logbook parts (one per container) to the logbook, in time order.

    On a shared network volume each container writes its own part, because
    concurrent appends to one file there are last-writer-wins.  Lines already
    in the logbook are skipped, so merging twice is harmless.
    """
    present = (
        set(wave.logbook.read_text(encoding="utf-8").splitlines())
        if wave.logbook.is_file()
        else set()
    )
    lines = []
    for part in sorted(Path(args.parts).glob("*.md")):
        for line in part.read_text(encoding="utf-8").splitlines():
            if line.startswith("- `") and line not in present:
                lines.append(line)
                present.add(line)
    lines.sort(key=lambda line: line[3:22])  # "- `YYYY-MM-DD HH:MM:SS`", stable
    with open(wave.logbook, "a", encoding="utf-8") as handle:
        handle.writelines(f"{line}\n" for line in lines)
    print(f"appended {len(lines)} lines to {wave.logbook}")
    return 0


def _attempt_label(wave: Wave, attempts: list[str]) -> str:
    """FAIL only for an attempt the driver judged; a run with no marker yet is
    either still collecting or was cut off, and reads "incomplete"."""
    if not attempts:
        return ""
    status = common.job_status(wave.data_root, attempts[-1], "collect")
    return "FAIL" if status is not None else "incomplete"


def status_table(wave: Wave) -> pd.DataFrame:
    rows = []
    for stage in ("s0", "s1", "s1b", "s2"):
        for run in common.planned_runs(wave.plan, stage):
            attempts = common.attempt_ids(wave.data_root, run.run_id)
            done = common.completed_attempt(wave.data_root, run.run_id)
            row = {
                "stage": stage,
                "ckpt": run.ckpt,
                "key": run.key,
                "attempts": len(attempts),
                "run_id": done or (attempts[-1] if attempts else ""),
                "collect": "ok" if done else _attempt_label(wave, attempts),
            }
            for kind in ("probe-pilot", "probe-linear", "probe-mlp"):
                status = done and common.job_status(wave.data_root, done, kind)
                row[kind] = ("ok" if status.get("ok") else "FAIL") if status else ""
            rows.append(row)
    return pd.DataFrame(rows)


def cmd_status(wave: Wave, args) -> int:
    print(status_table(wave).to_string(index=False))
    used = common.gpu_hours_used(wave.data_root)
    print(
        f"\nGPU (collection) hours: {used:.2f} / {wave.plan['budget']['wave_gpu_hours']}"
    )
    for stage in ("s0", "s0_envs20", "s1", "s1b"):
        for ckpt in common.CKPTS:
            gate = wave.gate(stage, ckpt)
            if gate:
                print(
                    f"gate {stage:9s} {ckpt:4s}: {'PASSED' if gate['passed'] else 'FAILED'}"
                )
    print(f"lock: {wave.state.get('lock')}")
    print(f"sizing: {wave.state.get('sizing')}")
    if wave.halt_path.exists():
        print(f"HALTED: {wave.halt_path.read_text().strip()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", default=str(common.PROBE_DIR / "data"))
    parser.add_argument("--logbook", default=str(common.PROBE_DIR / "logbook.md"))
    parser.add_argument("--plan", default=str(common.PLAN_PATH))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch a pinned checkpoint and verify it")
    p.add_argument("--ckpt", required=True, choices=common.CKPTS)
    p.add_argument("--dest", required=True)

    p = sub.add_parser("init", help="register and verify both checkpoints")
    p.add_argument("--sft", required=True)
    p.add_argument("--grpo", required=True)
    p.add_argument(
        "--no-hash", action="store_true", help="skip shard sha256 (not advised)"
    )
    p.add_argument("--headroom-gb", type=float, default=1.0)

    for name in ("s0", "s0-envs20"):
        p = sub.add_parser(
            name, help=f"{name} smoke on both checkpoints, then its gate"
        )
        p.add_argument("--ckpt", nargs="*", choices=common.CKPTS)

    p = sub.add_parser("lock", help="freeze num_envs / MUJOCO_GL / speed factor")
    p.add_argument("--num-envs", type=int, default=10)
    p.add_argument(
        "--speed-factor",
        type=float,
        default=1.0,
        help="this box's collection wall clock relative to the wave-2 card",
    )

    for name, text in (
        ("s1", "pilot collection + pilot probes + gates (registered gate)"),
        ("s1b", "Amendment 1 pilot2 on all goal tasks + amended gate"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("--ckpt", nargs="*", choices=common.CKPTS)
        p.add_argument("--jobs", type=int, default=None)

    sub.add_parser("size", help="freeze Amendment 1's GRPO episode counts")

    p = sub.add_parser("s2", help="main collections, linear probes behind them")
    p.add_argument("--ckpt", nargs="*", choices=common.CKPTS)
    p.add_argument("--only", nargs="*", help="plan keys, e.g. goal_swap_s1")
    p.add_argument(
        "--probes", choices=["background", "inline", "none"], default="background"
    )
    p.add_argument("--probe-jobs", type=int, default=None)

    p = sub.add_parser("s3", help="MLP probes for goal-suite runs (CPU pod)")
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument("--wait", action="store_true", help="keep polling for new runs")
    p.add_argument("--poll-s", type=float, default=120.0)
    p.add_argument("--retry-failed", action="store_true")

    p = sub.add_parser("collect", help="one planned collection (used by modal_e4.py)")
    p.add_argument("--ckpt", required=True, choices=common.CKPTS)
    p.add_argument("--key", required=True)

    p = sub.add_parser("probe", help="one probe round on a collected run")
    p.add_argument("--run-id", required=True)
    p.add_argument("--family", required=True, choices=["linear", "mlp", "pilot"])
    p.add_argument("--jobs", type=int, default=None)

    p = sub.add_parser("competence", help="per-task success / strength flags for a run")
    p.add_argument("--run-id", required=True)

    p = sub.add_parser("resume", help="clear a HALT after review")
    p.add_argument("--reason", required=True)

    sub.add_parser("todo", help="runnable S2 collections and probe rounds, as JSON")
    p = sub.add_parser("merge-logbook", help="append per-container logbook parts")
    p.add_argument("--parts", required=True, help="directory of *.md logbook parts")

    sub.add_parser("status")
    return parser


COMMANDS = {
    "download": cmd_download,
    "init": cmd_init,
    "s0": cmd_s0,
    "s0-envs20": lambda wave, args: cmd_s0(wave, args, stage="s0_envs20"),
    "lock": cmd_lock,
    "s1": cmd_s1,
    "s1b": lambda wave, args: cmd_s1(wave, args, stage="s1b"),
    "size": cmd_size,
    "s2": cmd_s2,
    "s3": cmd_s3,
    "collect": cmd_collect,
    "probe": cmd_probe,
    "competence": cmd_competence,
    "resume": cmd_resume,
    "todo": cmd_todo,
    "merge-logbook": cmd_merge_logbook,
    "status": cmd_status,
}


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    wave = Wave(
        Path(args.data_root).resolve(),
        Path(args.logbook),
        common.load_plan(Path(args.plan)),
    )
    wave.data_root.mkdir(parents=True, exist_ok=True)
    try:
        return COMMANDS[args.command](wave, args)
    except StageError as error:
        print(f"E4: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
