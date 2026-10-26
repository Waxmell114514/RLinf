"""Copy the E4 Volume from an old Modal workspace to the current one.

No container runs: every byte passes through this client, read with the old
workspace's token and written with the current one, so the copy costs no
Modal compute.  It is resumable (a file already in the new Volume at the same
size is skipped) and ends by comparing the two listings file by file, exiting
1 on any difference.  Nothing in the old Volume is changed or deleted.

    MODAL_OLD_TOKEN_ID / MODAL_OLD_TOKEN_SECRET   a token of the old workspace
    MODAL_TOKEN_ID / MODAL_TOKEN_SECRET           the current workspace

    python experiments/efference_probe/e4_copy_volume.py --dry-run
    python experiments/efference_probe/e4_copy_volume.py

After the copy, re-register the checkpoints (``modal_e4.py::stage --cmd "init
--sft /vol/ckpts/sft --grpo /vol/ckpts/grpo"``): the driver state records them
under the old Volume's mount path, and ``init`` re-checks their sha256s.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

VOLUME = "efference-e4"


def listing(volume) -> dict[str, int]:
    """Every file in the Volume, as path -> size."""
    import modal

    return {
        "/" + entry.path.lstrip("/"): entry.size
        for entry in volume.listdir("/", recursive=True)
        if entry.type == modal.volume.FileEntryType.FILE
    }


def to_copy(
    old: dict[str, int], new: dict[str, int], skip: tuple[str, ...]
) -> list[str]:
    """Old files the new Volume lacks or holds at another size, in path order."""
    return sorted(
        path
        for path, size in old.items()
        if not path.startswith(skip) and new.get(path) != size
    )


def unit(path: str) -> str:
    """The upload batch a file belongs to: a run directory or a checkpoint."""
    parts = path.strip("/").split("/")
    return "/".join(parts[:2]) if len(parts) > 2 else parts[0]


def differences(
    old: dict[str, int], new: dict[str, int], skip: tuple[str, ...]
) -> list[str]:
    wanted = {p: s for p, s in old.items() if not p.startswith(skip)}
    out = [f"missing {p}" for p in sorted(wanted) if p not in new]
    out += [
        f"size {p}: old {wanted[p]} new {new[p]}"
        for p in sorted(wanted)
        if p in new and new[p] != wanted[p]
    ]
    return out


def _client(prefix: str):
    import modal

    token_id = os.environ.get(f"{prefix}_TOKEN_ID")
    secret = os.environ.get(f"{prefix}_TOKEN_SECRET")
    if not token_id or not secret:
        raise SystemExit(f"{prefix}_TOKEN_ID / {prefix}_TOKEN_SECRET are not set")
    return modal.Client.from_credentials(token_id, secret)


def main(argv: list[str] | None = None) -> int:
    import modal

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--old-prefix", default="MODAL_OLD")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="path prefixes not to copy, e.g. /ckpts (then download them again)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    skip = tuple(args.skip)

    old_vol = modal.Volume.from_name(VOLUME, client=_client(args.old_prefix))
    new_vol = modal.Volume.from_name(
        VOLUME, create_if_missing=True, client=_client("MODAL")
    )
    old_vol.hydrate()
    new_vol.hydrate()
    if old_vol.object_id == new_vol.object_id:
        raise SystemExit("both tokens reach the same Volume; nothing to copy")

    old, new = listing(old_vol), listing(new_vol)
    pending = to_copy(old, new, skip)
    batches: dict[str, list[str]] = defaultdict(list)
    for path in pending:
        batches[unit(path)].append(path)
    total = sum(old[p] for p in pending)
    print(
        f"old: {len(old)} files {sum(old.values()) / 1e9:.2f} GB; already in new: "
        f"{len(old) - len(pending)}; to copy: {len(pending)} files {total / 1e9:.2f} GB "
        f"in {len(batches)} batches",
        flush=True,
    )
    if args.dry_run:
        for name, paths in sorted(batches.items()):
            print(
                f"  {name}: {len(paths)} files {sum(old[p] for p in paths) / 1e9:.2f} GB"
            )
        return 0

    for name, paths in sorted(batches.items()):
        size = sum(old[p] for p in paths)
        free = shutil.disk_usage(tempfile.gettempdir()).free
        if size > free - 2e9:
            raise SystemExit(
                f"{name}: {size / 1e9:.1f} GB does not fit in local temp space"
            )
        started = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            with new_vol.batch_upload(force=True) as batch:
                for path in paths:
                    local = Path(tmp) / path.lstrip("/")
                    local.parent.mkdir(parents=True, exist_ok=True)
                    with open(local, "wb") as handle:
                        for chunk in old_vol.read_file(path):
                            handle.write(chunk)
                    batch.put_file(str(local), path)
        print(
            f"{name}: {len(paths)} files {size / 1e9:.2f} GB in {time.time() - started:.0f} s",
            flush=True,
        )

    problems = differences(old, listing(new_vol), skip)
    for line in problems[:50]:
        print(line)
    print(
        "copy verified: every file present at its old size"
        if not problems
        else f"{len(problems)} differences",
        flush=True,
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
