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
    modal run --detach $M::wave                       # S2 + linear + S3 MLP, unattended
    modal run $M::stage --cmd "status"
    modal run $M::pull                                # analysis dirs + logbook parts, no hidden/

``E4_GPU`` picks the card (default ``L40S``; ``A10`` / ``L4`` are the 24 GB
options).  Set it identically for every command of the wave -- the lock
records the GPU, and the expected wall clocks assume one card type.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

GPU = os.environ.get("E4_GPU", "L40S")
REPO_LOCAL = Path(__file__).resolve().parents[2]
REPO = "/root/RLinf"
VOL = "/vol"
DATA = f"{VOL}/data"
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
        f"uv pip install --python {PYTHON} scikit-learn==1.9.1 pandas==2.3.3 joblib==1.6.0 "
        "pyarrow matplotlib pytest huggingface_hub pyyaml",
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
# for num_envs=10 (~40 GB measured), limit above it so a spike is not an OOM.
@app.function(
    gpu=GPU, cpu=(8.0, 12.0), memory=(49152, 65536), timeout=4 * 3600, **COMMON
)
def gpu_stage(args: list[str], mujoco_gl: str = "egl") -> int:
    return _driver(
        args, f"gpu_{args[0]}", {"MUJOCO_GL": mujoco_gl, "PYOPENGL_PLATFORM": mujoco_gl}
    )


@app.function(
    gpu=GPU, cpu=(8.0, 12.0), memory=(49152, 65536), timeout=14 * 3600, **COMMON
)
def gpu_lane() -> list[dict]:
    """S2: every pending collection, one at a time, committed after each.

    A preempted container re-runs this from the top; the driver then sees the
    interrupted attempt and collects under a new id.
    """
    results = []
    while True:
        todo = _todo()
        if todo["halted"] or not todo["collect"]:
            break
        ckpt, key = todo["collect"][0]
        rc = _driver(["collect", "--ckpt", ckpt, "--key", key], f"collect_{ckpt}_{key}")
        results.append({"ckpt": ckpt, "key": key, "rc": rc})
        if rc != 0:
            break
    return results


# 8 physical cores = 16 vCPU and 32 GB: the wave-2 CPU pod, whose MLP wall
# clocks (~2.2 h/run) the plan's expected_s assumes.
@app.function(cpu=8.0, memory=32768, timeout=6 * 3600, **COMMON)
def cpu_stage(args: list[str]) -> int:
    return _driver(args, "cpu_" + "_".join(a.lstrip("-") for a in args[:3]))


@app.function(cpu=0.25, memory=1024, timeout=24 * 3600, **COMMON)
def orchestrate(poll_s: int = 120) -> dict:
    """Start the GPU lane, then probe each run as soon as it lands."""
    lane = gpu_lane.spawn()
    calls: dict[tuple[str, str], modal.FunctionCall] = {}

    def finished(call: modal.FunctionCall) -> bool:
        try:
            call.get(timeout=0)
            return True
        except modal.exception.TimeoutError:
            return False
        except Exception:  # noqa: BLE001 - a failed call is finished; its marker says how
            return True

    while True:
        todo = _todo()
        for family in ("linear", "mlp"):
            for run_id in todo[family]:
                if (run_id, family) not in calls:
                    calls[(run_id, family)] = cpu_stage.spawn(
                        ["probe", "--run-id", run_id, "--family", family]
                    )
        # Done when nothing runs and nothing new is on offer.  A job still
        # offered after its call finished died before writing a marker (e.g.
        # OOM); it is not respawned -- a human decides.
        if (
            finished(lane)
            and all(finished(c) for c in calls.values())
            and all((r, f) in calls for f in ("linear", "mlp") for r in todo[f])
        ):
            break
        time.sleep(poll_s)
    return {"probes": sorted(f"{r}:{f}" for r, f in calls), "todo_at_exit": todo}


GPU_COMMANDS = {"s0", "s0-envs20", "s1", "collect"}


@app.local_entrypoint()
def stage(cmd: str, mujoco_gl: str = "egl"):
    args = cmd.split()
    if args[0] == "s0-envs20":
        # ~79 GB peak RSS at 20 envs on the wave-2 cluster run.
        rc = gpu_stage.with_options(memory=(98304, 131072)).remote(args, mujoco_gl)
    elif args[0] in GPU_COMMANDS:
        rc = gpu_stage.remote(args, mujoco_gl)
    else:
        rc = cpu_stage.remote(args)
    print(f"e4_wave.py {cmd} -> rc={rc}")


@app.local_entrypoint()
def wave():
    print(json.dumps(orchestrate.remote(), indent=2))


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
