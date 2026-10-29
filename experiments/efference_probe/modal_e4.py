"""Run the E4 wave on Modal: one GPU lane, parallel CPU probe lanes, one Volume.

Untested against a live Modal account (written against the ``modal`` 1.6.2
client API).  All experiment logic lives in ``e4_wave.py``; this file only
decides which container a driver command runs in.  From the repo root::

    pip install modal && modal setup
    M=experiments/efference_probe/modal_e4.py
    modal run $M::stage --cmd "download --ckpt sft --dest /vol/ckpts/sft"
    modal run $M::stage --cmd "download --ckpt grpo --dest /vol/ckpts/grpo"
    modal run $M::stage --cmd "init --sft /vol/ckpts/sft --grpo /vol/ckpts/grpo"
    modal run $M::stage --cmd "s0"                    # GPU; add --mujoco-gl osmesa if EGL fails
    modal run $M::stage --cmd "s0-envs20"             # optional, GPU with 96 GiB RAM
    modal run $M::stage --cmd "lock --num-envs 10 --speed-factor 1.0"
    modal run $M::stage --cmd "s1"                    # GPU
    modal run $M::stage --cmd "s1b"                   # GPU; Amendment 1 pilot2 + gate
    modal run $M::stage --cmd "size"                  # CPU; freezes GRPO's S2 sizing
    modal run --detach $M::wave --budget-usd 30       # S2 + linear + S3 MLP, unattended
    modal run $M::stage --cmd "status"
    modal run $M::pull                                # analysis dirs + logbook parts, no hidden/

``E4_GPU`` picks the card (default ``L40S``; ``A10`` / ``L4`` are the 24 GB
options).  Set it identically for every command of the wave -- the lock
records the GPU, and the expected wall clocks assume one card type.

``wave`` spends at most ``--budget-usd``: before starting a round it reserves
the round's worst case (its watchdog limit, priced at the container's resource
limits) and holds the round, and every round queued after it, if the
reservations would pass the budget.  Set the budget below what the Modal
workspace's spend limit still allows -- a workspace stop kills every running
container, and an MLP round that is cut off leaves nothing behind.  Spend is
journalled per session under ``data/e4_spend/`` (one file per event).
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import modal

GPU = os.environ.get("E4_GPU", "L40S")
REPO = "/root/RLinf"


def _repo_local(module_file: str, local: bool) -> Path:
    """The repo root on the client; in a container, the mounted copy.

    Modal imports this file as ``/root/modal_e4.py`` inside the container,
    which has no ``parents[2]``; the image spec is not rebuilt there.
    """
    return Path(module_file).resolve().parents[2] if local else Path(REPO)


REPO_LOCAL = _repo_local(__file__, modal.is_local())

# The install leaves numpy 1.26.4 and an unpinned pyarrow (26.x), which
# refuses to import under numpy < 2; every parquet write then fails.
ANALYSIS_PINS = (
    "scikit-learn==1.9.1",
    "pandas==2.3.3",
    "joblib==1.6.0",
    "pyarrow<20",
    "matplotlib",
    "pytest",
    "huggingface_hub",
    "pyyaml",
)
VOL = "/vol"
DATA = f"{VOL}/data"
SPEND = f"{DATA}/e4_spend"
INSTALL = "/opt/rlinf"
PYTHON = f"{INSTALL}/.venv/bin/python"
DRIVER = f"{REPO}/experiments/efference_probe/e4_wave.py"

app = modal.App("efference-e4")
volume = modal.Volume.from_name("efference-e4", create_if_missing=True)

# The install layer depends only on pyproject.toml + requirements/, as in the
# repo's Dockerfile; code is mounted last so editing it does not reinstall.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install(
        "git",
        "build-essential",
        "cmake",
        "pkg-config",
        "wget",
        "curl",
        "unzip",
        "sudo",
        # Wave 2: the GPU box shipped no glvnd libEGL; osmesa is the fallback.
        "libegl1",
        "libgl1",
        "libglvnd0",
        "libgles2",
        "libglx0",
        "libosmesa6",
        "libglew2.2",
        "libglfw3",
    )
    .env(
        {
            # No GPU at build time: pin the wheel family wave 2 ran (torch
            # 2.11.0+cu130); Modal's host driver is 580.x / CUDA 13.0.
            "UV_TORCH_BACKEND": "cu130",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
        }
    )
    .add_local_file(
        REPO_LOCAL / "pyproject.toml", f"{INSTALL}/pyproject.toml", copy=True
    )
    .add_local_dir(REPO_LOCAL / "requirements", f"{INSTALL}/requirements", copy=True)
    .run_commands(
        "python -m pip install --upgrade uv",
        f"cd {INSTALL} && bash requirements/install.sh embodied --model openvla-oft --env libero",
        # Analysis extras at the wave-2 versions.
        f"uv pip install --python {PYTHON} {' '.join(map(shlex.quote, ANALYSIS_PINS))}",
    )
    .add_local_dir(
        REPO_LOCAL,
        REPO,
        copy=False,
        ignore=[
            ".git",
            "**/__pycache__",
            "experiments/efference_probe/data",
            "**/*.pdf",
            "docs",
        ],
    )
)

COMMON = {"image": image, "volumes": {VOL: volume}}


def _driver(args: list[str], label: str, env: dict[str, str] | None = None) -> int:
    """One ``e4_wave.py`` command against the volume, committed afterwards.

    Each container writes its own logbook part; ``pull`` merges them into the
    repo's logbook (concurrent appends to one Volume file are last-writer-wins).
    """
    volume.reload()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    logbook = f"{DATA}/e4_logbook/{stamp}_{label}.md"
    Path(logbook).parent.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(
        [PYTHON, DRIVER, "--data-root", DATA, "--logbook", logbook, *args],
        env={**os.environ, **(env or {})},
        check=False,
    ).returncode
    volume.commit()
    return rc


def _part_label(prefix: str, args: list[str]) -> str:
    """A flat logbook-part name: ``init --sft /vol/ckpts/sft`` once made a
    nested path that ``merge-logbook``'s ``*.md`` glob never saw."""
    args = _command_args(args)
    words = (a.lstrip("-").strip("/").replace("/", "_") for a in args[:3])
    return "_".join([prefix, *words])


GLOBAL_OPTIONS = ("--data-root", "--logbook", "--plan")


def _command_args(args: list[str]) -> list[str]:
    """``args`` from the subcommand on, past any leading global options."""
    i = 0
    while i < len(args) and args[i] in GLOBAL_OPTIONS:
        i += 2
    return list(args[i:])


def wave_args(wave: str, args: list[str]) -> list[str]:
    """Route an ``e4_wave.py`` command to a later wave's plan and data root.

    X1 and X4 (PREREG_X1_X4.md) reuse this driver and launcher with their own
    plan (``<wave>_plan.yaml``) and their own data root on the same Volume
    (``/vol/data/<wave>``), so nothing of E4's state is read or written.
    """
    if wave == "e4":
        return list(args)
    plan = f"{REPO}/experiments/efference_probe/{wave}_plan.yaml"
    return ["--data-root", f"{DATA}/{wave}", "--plan", plan, *args]


def _todo() -> dict:
    volume.reload()
    out = subprocess.run(
        [PYTHON, DRIVER, "--data-root", DATA, "--logbook", "/tmp/todo.md", "todo"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)


# 12 physical cores = 24 vCPU, the wave-2 GPU box's 23.8-core quota; RAM sized
# for num_envs=10 (~40 GB measured).  On Modal the pilots' tree RSS reached
# 61-62.5 GB and grew with call count, so the limit sits well above it; the
# request (what is billed when unused) is unchanged.
GPU_CPU = (8.0, 12.0)
GPU_MEMORY = (49152, 98304)
# One Modal core is one CPU inside the container: a bare ``cpu=8.0`` gave the
# old workspace's MLP rounds 24 CPUs, and a 12-core limit (b4963f0) halved
# them and ran the GRPO MLP retry ~3x slower.  MLP rounds peaked at 27.9 GB
# tree RSS with 24 workers.  Modal preempted MLP rounds four times and each
# restarted from scratch, so the remaining ones run off Modal (LOCAL_RUN.md).
PROBE_CPU = (8.0, 24.0)
PROBE_MEMORY = (32768, 49152)
ORCHESTRATOR_CPU, ORCHESTRATOR_MEMORY = 0.25, 1024

# Modal list prices (USD), October 2026.  Modal bills the greater of request
# and use, so the budget gate prices each container at its limits.
GPU_USD_PER_H = {"L40S": 1.951, "A10": 1.102, "L4": 0.799, "H100": 3.949}
CORE_USD_PER_S = 0.0000131  # per physical core (2 vCPU)
GIB_USD_PER_S = 0.00000222
# Container start before the driver's clock runs; reserved on top of max_s.
STARTUP_S = 300


def usd_per_hour(cores: float, memory_mib: float, gpu: Optional[str] = None) -> float:
    return 3600 * (cores * CORE_USD_PER_S + memory_mib / 1024 * GIB_USD_PER_S) + (
        GPU_USD_PER_H[gpu] if gpu else 0.0
    )


USD_PER_H = {
    "collect": usd_per_hour(GPU_CPU[1], GPU_MEMORY[1], GPU),
    "linear": usd_per_hour(PROBE_CPU[1], PROBE_MEMORY[1]),
    "mlp": usd_per_hour(PROBE_CPU[1], PROBE_MEMORY[1]),
    "orchestrate": usd_per_hour(ORCHESTRATOR_CPU, ORCHESTRATOR_MEMORY),
}


@app.function(gpu=GPU, cpu=GPU_CPU, memory=GPU_MEMORY, timeout=4 * 3600, **COMMON)
def gpu_stage(args: list[str], mujoco_gl: str = "egl") -> int:
    return _driver(
        args,
        _part_label("gpu", args),
        {"MUJOCO_GL": mujoco_gl, "PYOPENGL_PLATFORM": mujoco_gl},
    )


# 8 physical cores = 16 vCPU and 32 GB: the wave-2 CPU pod, whose MLP wall
# clocks (~2.2 h/run) the plan's expected_s assumes.
@app.function(cpu=PROBE_CPU, memory=PROBE_MEMORY, timeout=6 * 3600, **COMMON)
def cpu_stage(args: list[str]) -> int:
    return _driver(args, _part_label("cpu", args))


# A preempted MLP round restarts from zero (it cannot checkpoint); this one is
# never preempted, at Modal's 3x price for CPU and memory.
@app.function(
    cpu=PROBE_CPU,
    memory=PROBE_MEMORY,
    timeout=6 * 3600,
    nonpreemptible=True,
    **COMMON,
)
def cpu_stage_nonpreemptible(args: list[str]) -> int:
    return _driver(args, _part_label("cpu_np", args))


# Modal restarts a preempted input from the top, and each restart of an MLP
# round costs a full round again; past the cap the round refuses to run.
MAX_PROBE_ATTEMPTS = 2


def claim_attempt(path: Path, cap: int = MAX_PROBE_ATTEMPTS) -> Optional[int]:
    """Count one more attempt in ``path``; None (and no count) once ``cap`` is used."""
    seen = json.loads(path.read_text())["attempts"] if path.is_file() else []
    if len(seen) >= cap:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(json.dumps({"attempts": [*seen, stamp]}))
    return len(seen) + 1


# 8 h: room for the driver's watchdog at --speed-factor 1.5 (7 h for MLP).
@app.function(cpu=PROBE_CPU, memory=PROBE_MEMORY, timeout=8 * 3600, **COMMON)
def probe_stage(
    run_id: str, family: str, launch: int = 1, speed_factor: float = 1.0
) -> int:
    """One probe round, preemptible, at most ``MAX_PROBE_ATTEMPTS`` starts.

    ``launch`` > 1 is a deliberate relaunch (logged in the logbook) with its
    own attempt count; earlier launches' counts stay on the Volume.
    ``speed_factor`` is the driver's: it scales the expected time and watchdog.
    """
    label = _part_label("probe", [run_id, family]) + (
        f"_launch{launch}" if launch > 1 else ""
    )
    volume.reload()
    attempt = claim_attempt(Path(DATA) / "e4_attempts" / f"{label}.json")
    volume.commit()
    if attempt is None:
        print(f"{family} {run_id}: {MAX_PROBE_ATTEMPTS} attempts used; not starting")
        return 75
    print(f"{family} {run_id}: attempt {attempt} of {MAX_PROBE_ATTEMPTS}", flush=True)
    return _driver(
        [
            "probe",
            "--run-id",
            run_id,
            "--family",
            family,
            "--speed-factor",
            f"{speed_factor:g}",
        ],
        f"{label}_a{attempt}",
    )


def _outcome(call) -> Optional[str]:
    """None while a spawned call runs; once it has returned, failed or expired, how.

    Known gap: after a preemption Modal can mark the call FAILURE in its call
    graph while ``get(timeout=0)`` still raises ``TimeoutError``, so such a
    call reads as running and keeps its slot (seen on the second wave).

    modal 1.6 raises the *builtin* ``TimeoutError`` for a call still running
    under ``get(timeout=0)``, not ``modal.exception.TimeoutError``; catching
    only the latter read a running GPU lane as finished on the first poll.
    """
    try:
        return f"rc={call.get(timeout=0)}"
    except modal.exception.OutputExpiredError:
        return "expired"
    except (TimeoutError, modal.exception.TimeoutError):
        return None
    except Exception as error:  # noqa: BLE001 - a failed call is finished; its marker says how
        return f"error: {type(error).__name__}"


# -- budget ------------------------------------------------------------------

DEFAULT_MAX_RUNNING = {"collect": 1, "linear": 2, "mlp": 2}


def reserve_usd(item: dict[str, Any], rates: dict[str, float] = USD_PER_H) -> float:
    """A round's worst case: its watchdog limit plus start-up, at limit rates."""
    return rates[item["kind"]] * (item["max_s"] + STARTUP_S) / 3600


def journal_usd(entries: list[dict[str, Any]], now: float) -> tuple[float, float]:
    """``(spent, committed)`` for one session's spend journal.

    Each container is timed from its spawn to the poll that saw it return (or
    to ``now``) at its kind's limit rate, so ``spent`` over-states the bill.
    ``committed`` adds, for every container still running, the rest of its
    reservation.
    """
    ends = {e["key"]: e for e in entries if e["event"] == "end"}
    spent = committed = 0.0
    for start in (e for e in entries if e["event"] == "start"):
        end = ends.get(start["key"])
        cost = start["usd_per_h"] * ((end["t"] if end else now) - start["t"]) / 3600
        spent += cost
        reserved = start["usd_per_h"] * start["reserve_s"] / 3600
        committed += cost if end else max(cost, reserved)
    return spent, committed


def pick(
    queue: list[dict[str, Any]],
    taken: set[str],
    running: Counter,
    committed: float,
    budget_usd: float,
    max_running: dict[str, int],
    rates: dict[str, float] = USD_PER_H,
) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
    """The rounds to start now, in queue order, and the round the budget holds.

    A round whose kind is at its cap is passed over, so a round of another
    kind may start.  A round whose reservation does not fit the budget holds
    itself and everything queued after it.
    """
    chosen = []
    counts = Counter(running)
    for item in queue:
        kind = item["kind"]
        if item["key"] in taken or counts[kind] >= max_running.get(kind, 0):
            continue
        cost = reserve_usd(item, rates)
        if committed + cost > budget_usd:
            return chosen, item
        chosen.append(item)
        counts[kind] += 1
        committed += cost
    return chosen, None


def _journal(session: str) -> list[dict[str, Any]]:
    root = Path(SPEND) / session
    if not root.is_dir():
        return []
    return [json.loads(path.read_text()) for path in sorted(root.glob("*.json"))]


def _journal_add(session: str, event: dict[str, Any]) -> None:
    """One file per event, committed at once so a later ``reload()`` keeps it."""
    root = Path(SPEND) / session
    root.mkdir(parents=True, exist_ok=True)
    name = f"{event['t']:.3f}_{event['event']}_{event['key'].replace(':', '_')}.json"
    (root / name).write_text(json.dumps(event))
    volume.commit()


def _note(session: str, message: str) -> None:
    """A line in this session's logbook part (merged by ``pull``)."""
    path = Path(DATA) / "e4_logbook" / f"{session}_orchestrate.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"- `{stamp}` **E4 (orchestrator)** {message}\n")
    volume.commit()


@app.function(
    cpu=ORCHESTRATOR_CPU, memory=ORCHESTRATOR_MEMORY, timeout=24 * 3600, **COMMON
)
def orchestrate(
    budget_usd: float,
    session: str,
    max_running: Optional[dict[str, int]] = None,
    skip: tuple[str, ...] = (),
    rates: Optional[dict[str, float]] = None,
    poll_s: int = 120,
) -> dict:
    """Start pending rounds in ``todo``'s queue order, within caps and budget.

    Collections run one at a time on the GPU (the first failed one stops
    collections for the session, as the old lane did); probe rounds run beside
    them as soon as their run lands.  A round that returns without a marker
    (e.g. OOM) is not restarted -- a human decides.  ``skip`` lists
    ``kind:run_id`` rounds running elsewhere: a second ``probe`` on the same
    run would rename the live analysis dir.  A restarted orchestrator with the
    same ``session`` re-reads the journal and re-attaches to its rounds.
    ``rates`` come from the client, whose ``E4_GPU`` picked the card; this
    container does not see it.
    """
    caps = {**DEFAULT_MAX_RUNNING, **(max_running or {})}
    rates = rates or USD_PER_H
    volume.reload()
    entries = _journal(session)
    closed = {e["key"] for e in entries if e["event"] == "end"}
    calls: dict[str, Any] = {}
    for start in entries:
        if start["event"] != "start" or start["key"] in closed:
            continue
        if start["kind"] == "orchestrate":
            # An earlier orchestrator of this session died; it stops costing now.
            _journal_add(
                session,
                {
                    "event": "end",
                    "key": start["key"],
                    "t": time.time(),
                    "outcome": "replaced",
                },
            )
            continue
        try:
            calls[start["key"]] = modal.FunctionCall.from_id(start["call_id"])
        except Exception as error:  # noqa: BLE001
            _journal_add(
                session,
                {
                    "event": "end",
                    "key": start["key"],
                    "t": time.time(),
                    "outcome": f"lost: {type(error).__name__}",
                },
            )
    taken = {e["key"] for e in entries if e["event"] == "start"} | set(skip)
    me = f"orchestrate:{time.time():.0f}"
    _journal_add(
        session,
        {
            "event": "start",
            "key": me,
            "kind": "orchestrate",
            "t": time.time(),
            "usd_per_h": rates["orchestrate"],
            "reserve_s": 0,
        },
    )
    _note(
        session,
        f"session `{session}` up: budget ${budget_usd:.2f}, caps {caps}, re-attached {sorted(calls)}",
    )

    held_key, collect_failed, todo = None, False, {}
    while True:
        for key, call in list(calls.items()):
            outcome = _outcome(call)
            if outcome is None:
                continue
            del calls[key]
            _journal_add(
                session,
                {"event": "end", "key": key, "t": time.time(), "outcome": outcome},
            )
            _note(session, f"`{key}` returned: {outcome}")
            if key.startswith("collect:") and outcome != "rc=0":
                collect_failed = True
                _note(session, "a collection failed; no more collections this session")
        todo = _todo()
        queue = [
            item
            for item in todo["queue"]
            if not (collect_failed and item["kind"] == "collect")
        ]
        running = Counter(key.split(":", 1)[0] for key in calls)
        _, committed = journal_usd(_journal(session), time.time())
        chosen, held = pick(queue, taken, running, committed, budget_usd, caps, rates)
        for item in chosen:
            stage_fn = gpu_stage if item["kind"] == "collect" else cpu_stage
            call = stage_fn.spawn(item["args"])
            calls[item["key"]] = call
            taken.add(item["key"])
            _journal_add(
                session,
                {
                    "event": "start",
                    "key": item["key"],
                    "kind": item["kind"],
                    "call_id": call.object_id,
                    "t": time.time(),
                    "usd_per_h": rates[item["kind"]],
                    "reserve_s": item["max_s"] + STARTUP_S,
                },
            )
            committed += reserve_usd(item, rates)
            _note(
                session,
                f"started `{item['key']}`: reserves ${reserve_usd(item, rates):.2f}, committed ${committed:.2f} of ${budget_usd:.2f}",
            )
        if held is not None and held["key"] != held_key:
            held_key = held["key"]
            _note(
                session,
                f"budget holds `{held_key}` and everything queued after it: it reserves ${reserve_usd(held, rates):.2f}, ${committed:.2f} of ${budget_usd:.2f} committed",
            )
        # A lane running elsewhere may still land runs to probe.
        external_lane = (
            caps.get("collect", 0) == 0 and todo["collect"] and not todo["halted"]
        )
        if not calls and not chosen and not external_lane:
            break
        time.sleep(poll_s)

    _journal_add(
        session, {"event": "end", "key": me, "t": time.time(), "outcome": "done"}
    )
    spent, _ = journal_usd(_journal(session), time.time())
    left = [item["key"] for item in todo.get("queue", []) if item["key"] not in taken]
    _note(
        session,
        f"session `{session}` done: ~${spent:.2f} by the journal (limit rates, so above the bill); not started: {left or 'none'}",
    )
    return {
        "session": session,
        "spent_usd": spent,
        "not_started": left,
        "held": held_key,
    }


GPU_COMMANDS = {"s0", "s0-envs20", "s1", "s1b", "collect"}


@app.local_entrypoint()
def stage(cmd: str, mujoco_gl: str = "egl", wave: str = "e4"):
    """One driver command; ``--wave x1`` / ``--wave x4`` for the later waves."""
    args = cmd.split()
    routed = wave_args(wave, args)
    if args[0] == "s0-envs20":
        # ~79 GB peak RSS at 20 envs on the wave-2 cluster run.
        rc = gpu_stage.with_options(memory=(98304, 131072)).remote(routed, mujoco_gl)
    elif args[0] in GPU_COMMANDS:
        rc = gpu_stage.remote(routed, mujoco_gl)
    else:
        rc = cpu_stage.remote(routed)
    print(f"[{wave}] e4_wave.py {cmd} -> rc={rc}")


@app.local_entrypoint()
def wave(
    budget_usd: float,
    lane: bool = True,
    skip: str = "",
    max_mlp: int = DEFAULT_MAX_RUNNING["mlp"],
    max_linear: int = DEFAULT_MAX_RUNNING["linear"],
    session: str = "",
):
    """Spawn the orchestrator and return; run it with ``modal run --detach``.

    Spawned, not ``.remote()``: a detached ``.remote()`` input is cancelled when
    the local client dies.  ``--budget-usd`` is the most this session may
    spend; keep it below the workspace's remaining Modal limit.  ``--no-lane``
    probes behind a GPU lane that is already running; ``--skip
    kind:run_id,...`` names rounds in flight elsewhere.  ``--session NAME``
    continues that session's journal (its spend counts against the budget).
    """
    if budget_usd <= 0:
        raise SystemExit("--budget-usd must be positive")
    session = session or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    caps = {"collect": 1 if lane else 0, "linear": max_linear, "mlp": max_mlp}
    call = orchestrate.spawn(
        budget_usd=budget_usd,
        session=session,
        max_running=caps,
        skip=tuple(item for item in skip.split(",") if item),
        rates=USD_PER_H,
    )
    print(
        f"orchestrate spawned: {call.object_id}; session {session}, "
        f"budget ${budget_usd:.2f}, caps {caps}"
    )


@app.local_entrypoint()
def probe_round(
    run_id: str, family: str = "mlp", launch: int = 1, speed_factor: float = 1.0
):
    """Spawn one preemptible, attempt-capped probe round; use ``--detach``."""
    call = probe_stage.spawn(run_id, family, launch, speed_factor)
    print(
        f"{family} {run_id}: spawned {call.object_id} (max {MAX_PROBE_ATTEMPTS} attempts)"
    )


@app.local_entrypoint()
def probe_np(run_id: str, family: str = "mlp"):
    """Spawn one non-preemptible probe round and return; use ``--detach``."""
    call = cpu_stage_nonpreemptible.spawn(
        ["probe", "--run-id", run_id, "--family", family]
    )
    print(f"{family} {run_id}: spawned {call.object_id} (non-preemptible)")


@app.local_entrypoint()
def pull(dest: str = "experiments/efference_probe/data"):
    """Everything but hidden/ and frames/, then merge the logbook parts.

    hidden/ stays on the Volume until the paper's numbers are final (wave-2
    rule); fetch it explicitly with ``modal volume get`` if a new probe family
    is ever wanted.
    """
    import sys

    root = Path(dest)
    for entry in volume.listdir("/data", recursive=True):
        rel = entry.path.lstrip("/").removeprefix("data/")
        skip = any(f"/{part}/" in f"/{rel}" for part in ("hidden", "frames"))
        if entry.type != modal.volume.FileEntryType.FILE or skip:
            continue
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as handle:
            for chunk in volume.read_file(entry.path):
                handle.write(chunk)
    # Needs the analysis stack (numpy/pandas/pyyaml) in the local interpreter.
    subprocess.run(
        [
            sys.executable,
            str(REPO_LOCAL / "experiments/efference_probe/e4_wave.py"),
            "--data-root",
            str(root),
            "merge-logbook",
            "--parts",
            str(root / "e4_logbook"),
        ],
        check=True,
    )
