"""Run E4's remaining CPU-only probe rounds on a local machine.

The goal-suite MLP rounds and the libero_long GRPO linear re-run need no GPU.
On Modal they were preempted and restarted from scratch (an MLP round cannot
checkpoint), so they run here instead -- on Windows, Linux or macOS --
through the unchanged driver (``e4_wave.py probe``: same gates, completion
markers, watchdog and logbook lines).  Data moves by client-side Volume reads
and writes; no Modal container runs.  See LOCAL_RUN.md.

    python -X utf8 experiments/efference_probe/e4_local.py check
    python -X utf8 experiments/efference_probe/e4_local.py download
    python -X utf8 experiments/efference_probe/e4_local.py run      # resumable
    python -X utf8 experiments/efference_probe/e4_local.py upload
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from efference_probe import e4_common as common  # noqa: E402

DRIVER = HERE / "e4_wave.py"
VOLUME = "efference-e4"
REMOTE_ROOT = "data"
BULK = ("hidden", "frames")  # never uploaded; hidden/ downloaded only where probed
# Amendment 2 (logbook): the libero_long GRPO linear round its 1600 s
# watchdog stopped, re-run under the amended expected time.
RERUN = (("linear", "e4_grpo_long_swap_s0"),)
# The plan's probe times were measured with 24 workers; fewer scale them up.
REFERENCE_JOBS = 24
# MLP rounds peaked at 27.9 GB tree RSS with 24 workers.
GB_PER_WORKER, BASE_GB = 1.2, 6.0
PINNED = {
    "numpy": "1.26.4",
    "scipy": "1.17.1",
    "sklearn": "1.9.1",
    "pandas": "2.3.3",
    "joblib": "1.6.0",
    "pyarrow": "19.0.1",
}


# -- planning (pure) ----------------------------------------------------------


def is_bulk(rel: str) -> bool:
    return any(part in BULK for part in rel.split("/")[1:-1])


def run_of(rel: str) -> str:
    return rel.split("/", 1)[0]


def rounds_from(todo: dict[str, Any], done) -> list[tuple[str, str]]:
    """``(family, run_id)`` in the driver's queue order, then the re-runs.

    ``done(family, run_id)`` says whether a round already has a passing marker.
    Collections are left out: they need a GPU.
    """
    out = [
        (item["kind"], item["key"].split(":", 1)[1])
        for item in todo["queue"]
        if item["kind"] in ("linear", "mlp")
    ]
    out += [(f, r) for f, r in RERUN if (f, r) not in out and not done(f, r)]
    return [(f, r) for f, r in out if not done(f, r)]


def to_download(
    remote: dict[str, int], local: dict[str, int], hidden_runs: set[str]
) -> list[str]:
    """Remote files the local copy lacks or holds at another size.

    Everything but hidden/ and frames/, plus hidden/ of the runs to probe.
    """
    wanted = [
        rel
        for rel in remote
        if not is_bulk(rel)
        or (rel.split("/")[1] == "hidden" and run_of(rel) in hidden_runs)
    ]
    return sorted(rel for rel in wanted if local.get(rel) != remote[rel])


def to_upload(local: dict[str, int], remote: dict[str, int]) -> list[str]:
    """Local results the Volume lacks or holds at another size; never bulk data."""
    return sorted(
        rel
        for rel, size in local.items()
        if not is_bulk(rel) and remote.get(rel) != size
    )


def choose_jobs(cpus: int, ram_bytes: int, requested: Optional[int] = None) -> int:
    if requested:
        return requested
    by_memory = int((ram_bytes / 2**30 - BASE_GB) / GB_PER_WORKER)
    return max(1, min(cpus, REFERENCE_JOBS, by_memory))


def speed_for(jobs: int) -> float:
    return max(1.0, REFERENCE_JOBS / jobs)


# -- I/O ------------------------------------------------------------------------


def local_listing(root: Path) -> dict[str, int]:
    if not root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.name.endswith(".part")
    }


def _volume():
    import modal

    try:
        volume = modal.Volume.from_name(VOLUME)
        volume.hydrate()
    except Exception as error:  # noqa: BLE001 - say what to fix, not a stack
        raise SystemExit(
            f"cannot open Modal Volume {VOLUME!r}: {type(error).__name__}: {error}\n"
            "Log in to the workspace that holds it first: `python -m modal token set "
            "--token-id ... --token-secret ...` (or set MODAL_TOKEN_ID / "
            "MODAL_TOKEN_SECRET)."
        ) from None
    return volume


def remote_listing(volume) -> dict[str, int]:
    import modal

    prefix = f"{REMOTE_ROOT}/"
    return {
        entry.path.lstrip("/")[len(prefix) :]: entry.size
        for entry in volume.listdir(f"/{REMOTE_ROOT}", recursive=True)
        if entry.type == modal.volume.FileEntryType.FILE
        and entry.path.lstrip("/").startswith(prefix)
    }


def _fetch(volume, files: list[str], root: Path, remote: dict[str, int]) -> None:
    total, done, started = sum(remote[r] for r in files), 0, time.time()
    for index, rel in enumerate(files, 1):
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        with open(partial, "wb") as handle:
            for chunk in volume.read_file(f"/{REMOTE_ROOT}/{rel}"):
                handle.write(chunk)
        os.replace(partial, target)
        done += remote[rel]
        if index % 200 == 0 or index == len(files):
            print(
                f"  {index}/{len(files)} files, {done / 1e9:.2f}/{total / 1e9:.2f} GB, "
                f"{time.time() - started:.0f} s",
                flush=True,
            )


def _driver(args) -> list[str]:
    """The driver command line up to its subcommand."""
    cmd = [
        sys.executable,
        "-X",
        "utf8",
        str(DRIVER),
        "--data-root",
        str(args.data_root),
    ]
    return cmd + (["--plan", str(args.plan)] if args.plan else [])


def _driver_todo(args) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        out = subprocess.run(
            [*_driver(args), "--logbook", str(Path(tmp) / "todo.md"), "todo"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=_child_env(),
            check=True,
        ).stdout
    return json.loads(out.strip().splitlines()[-1])


def _child_env() -> dict[str, str]:
    return {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def _done(root: Path):
    def done(family: str, run_id: str) -> bool:
        status = common.job_status(root, run_id, f"probe-{family}")
        return bool(status and status.get("ok"))

    return done


def _rounds(args) -> list[tuple[str, str]]:
    todo = _driver_todo(args)
    if todo["collect"]:
        print(f"note: collections still pending (GPU, not run here): {todo['collect']}")
    return rounds_from(todo, _done(args.data_root))


def _machine(root: Path, requested: Optional[int]) -> dict[str, Any]:
    import psutil

    cpus = os.cpu_count() or 1
    ram = int(psutil.virtual_memory().total)
    jobs = choose_jobs(cpus, ram, requested)
    root.mkdir(parents=True, exist_ok=True)
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpus": cpus,
        "ram_gb": round(ram / 2**30, 1),
        "disk_free_gb": round(shutil.disk_usage(root).free / 2**30, 1),
        "jobs": jobs,
        "speed_factor": round(speed_for(jobs), 3),
    }


def _versions() -> dict[str, str]:
    import importlib

    out = {}
    for name in PINNED:
        try:
            out[name] = importlib.import_module(name).__version__
        except ImportError:
            out[name] = "missing"
    return out


def _note(part: Path, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    part.parent.mkdir(parents=True, exist_ok=True)
    with open(part, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"- `{stamp}` **E4 (local)** {message}\n")


# -- commands -------------------------------------------------------------------


def cmd_check(args) -> int:
    machine = _machine(args.data_root, args.jobs)
    print(json.dumps(machine, indent=2))
    versions = _versions()
    off = {k: v for k, v in versions.items() if v != PINNED[k]}
    print("packages:", versions)
    if off:
        print(f"WARNING: not the pinned versions {PINNED}: {off}")
    if machine["ram_gb"] < 24 or machine["cpus"] < 8:
        print(
            "WARNING: below 8 CPUs / 24 GB the MLP rounds may run past their watchdog"
        )
    remote = remote_listing(_volume())
    print(f"Volume {VOLUME!r}: {len(remote)} files under /{REMOTE_ROOT}")
    if (args.data_root / "e4_state.json").is_file():
        rounds = _rounds(args)
        print(f"{len(rounds)} rounds to run, in order:")
        for family, run_id in rounds:
            hidden = (args.data_root / run_id / "hidden").is_dir()
            print(
                f"  {family:6s} {run_id}{'' if hidden else '  (hidden/ not downloaded)'}"
            )
    else:
        print("no local data yet: run `download`")
    return 0


def cmd_download(args) -> int:
    volume, root = _volume(), args.data_root
    remote = remote_listing(volume)
    small = to_download(remote, local_listing(root), set())
    print(f"bookkeeping and results: {len(small)} files", flush=True)
    _fetch(volume, small, root, remote)
    rounds = _rounds(args)
    runs = {run_id for _, run_id in rounds}
    bulk = to_download(remote, local_listing(root), runs)
    size = sum(remote[r] for r in bulk)
    free = shutil.disk_usage(root).free
    if size > free - 5e9:
        raise SystemExit(f"hidden/ needs {size / 1e9:.1f} GB; {free / 1e9:.1f} GB free")
    print(
        f"hidden/ of {len(runs)} runs: {len(bulk)} files {size / 1e9:.2f} GB",
        flush=True,
    )
    _fetch(volume, bulk, root, remote)
    missing = to_download(remote, local_listing(root), runs)
    print("download verified" if not missing else f"{len(missing)} files still differ")
    return 1 if missing else 0


def cmd_run(args) -> int:
    root = args.data_root
    machine = _machine(root, args.jobs)
    speed = args.speed_factor or machine["speed_factor"]
    rounds = _rounds(args)
    part = (
        root / "e4_logbook" / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_local_"
        f"{re.sub(r'[^A-Za-z0-9_-]', '_', machine['host'])}.md"
    )
    _note(
        part,
        f"local probe session on {machine['platform']} (python {machine['python']}, "
        f"{machine['cpus']} CPUs, {machine['ram_gb']} GB RAM): --jobs {machine['jobs']}, "
        f"speed factor {speed:g}; packages {_versions()}; {len(rounds)} rounds: "
        + ", ".join(f"{f}:{r}" for f, r in rounds),
    )
    failed = []
    for index, (family, run_id) in enumerate(rounds, 1):
        if not (root / run_id / "hidden").is_dir():
            print(f"{run_id}: hidden/ not downloaded; run `download` first")
            failed.append(f"{family}:{run_id}")
            continue
        print(f"\n[{index}/{len(rounds)}] {family} {run_id}", flush=True)
        subprocess.run(
            [
                *_driver(args),
                "--logbook",
                str(part),
                "probe",
                "--run-id",
                run_id,
                "--family",
                family,
                "--jobs",
                str(machine["jobs"]),
                "--speed-factor",
                f"{speed:g}",
            ],
            env=_child_env(),
            check=False,
        )
        status = common.job_status(root, run_id, f"probe-{family}") or {}
        print(
            f"[{index}/{len(rounds)}] {family} {run_id}: "
            f"{'ok' if status.get('ok') else 'FAILED'} "
            f"({status.get('elapsed_s', 0) / 60:.1f} min)",
            flush=True,
        )
        if not status.get("ok"):
            failed.append(f"{family}:{run_id}")
    _note(part, f"local probe session done; failed: {failed or 'none'}")
    print(f"\nfailed: {failed or 'none'}; next: `upload`")
    return 1 if failed else 0


def cmd_upload(args) -> int:
    volume, root = _volume(), args.data_root
    files = to_upload(local_listing(root), remote_listing(volume))
    print(f"uploading {len(files)} files", flush=True)
    if files:
        with volume.batch_upload(force=True) as batch:
            for rel in files:
                batch.put_file(str(root / rel), f"/{REMOTE_ROOT}/{rel}")
    left = to_upload(local_listing(root), remote_listing(volume))
    print("upload verified" if not left else f"{len(left)} files still differ")
    return 1 if left else 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=HERE / "data")
    parser.add_argument("--plan", type=Path, default=None, help="default: the driver's")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("check", "download", "run", "upload"):
        p = sub.add_parser(name)
        p.add_argument("--jobs", type=int, default=None)
        p.add_argument("--speed-factor", type=float, default=None)
    args = parser.parse_args(argv)
    args.data_root = args.data_root.resolve()
    return {
        "check": cmd_check,
        "download": cmd_download,
        "run": cmd_run,
        "upload": cmd_upload,
    }[args.cmd](args)


if __name__ == "__main__":
    if not sys.flags.utf8_mode:
        # Windows defaults to the locale code page; the driver writes UTF-8.
        # A child, not os.execv: on Windows execv returns to the prompt early.
        sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))
    sys.exit(main())
