#!/usr/bin/env python
"""Collect hijack-labelled calls with hidden states.

Example::

    export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
    export ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
    export PYTHONPATH=$PWD:$PYTHONPATH
    python experiments/efference_probe/run_collect.py \
        --config experiments/efference_probe/configs/smoke.yaml \
        --set model.model_path=/abs/path/to/ckpt run_id=smoke01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from efference_probe.config import load_config  # noqa: E402
from efference_probe.harness import EfferenceHarness, Logbook  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a run YAML")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="dotted overrides, e.g. env.num_envs=4 model.do_sample=true",
    )
    parser.add_argument(
        "--logbook",
        default=str(Path(__file__).resolve().parent / "logbook.md"),
        help="append-only research logbook",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and print the config without loading the model",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.set)
    if args.dry_run:
        import yaml

        print(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        print(f"# config hash: {cfg.hash()}")
        return 0

    logbook = Logbook(args.logbook)
    logbook.write(
        f"**START** run_id=`{cfg.run_id}` config=`{args.config}` "
        f"hash=`{cfg.hash()}` overrides=`{' '.join(args.set) or 'none'}`"
    )
    harness = EfferenceHarness(cfg, logbook=logbook)
    try:
        path = harness.run()
    except Exception as error:  # noqa: BLE001 - the logbook must record failures
        logbook.write(
            f"**FAILED** run_id=`{cfg.run_id}`: {type(error).__name__}: {error}"
        )
        raise
    logbook.write(f"**DONE** run_id=`{cfg.run_id}` -> `{path}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
