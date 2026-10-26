"""Stand-in for ``run_collect.py`` in the offline E4 flow test.

Takes the same arguments, resolves them through the real ``load_config`` (so
a bad override still fails here), and writes a synthetic run where the real
collector would, with a manifest shaped like ``RunWriter``'s.  Selected by
``E4_COLLECT_SCRIPT``; ``FAKE_COLLECT_FAIL=1`` simulates a run that dies
after creating its directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efference_probe.config import load_config  # noqa: E402
from efference_probe.tests.synthetic import make_synthetic_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--logbook", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    out_dir = Path(cfg.out_root) / cfg.run_id
    with open(args.logbook, "a", encoding="utf-8") as handle:
        handle.write(
            f"- `{time.strftime('%Y-%m-%d %H:%M:%S')}` **START** run_id=`{cfg.run_id}` (fake)\n"
        )
    if os.environ.get("FAKE_COLLECT_FAIL") == "1":
        (out_dir / "hidden").mkdir(parents=True)
        return 1

    smoke = cfg.env.max_calls_per_episode <= 6
    n_tasks = len(cfg.env.task_ids) if cfg.env.task_ids is not None else 3
    make_synthetic_run(
        out_dir,
        n_tasks=n_tasks,
        episodes_per_task=min(cfg.env.init_states_per_task, 8),
        n_calls=cfg.env.max_calls_per_episode if smoke else 24,
        hidden_size=4096 if smoke else 64,
        layers=list(cfg.capture.layers) if smoke else None,
        hidden_signal=0.0,
        transform=cfg.hijack.transform,
        seed=cfg.env.seed,
        command_mean=0.0,
    )
    run_config = yaml.safe_load((out_dir / "run_config.yaml").read_text())
    run_config.update(run_id=cfg.run_id, config_hash=cfg.hash())
    run_config["config"]["hijack"]["transform"] = cfg.hijack.transform
    (out_dir / "run_config.yaml").write_text(
        yaml.safe_dump(run_config, sort_keys=False)
    )

    calls = pd.read_parquet(out_dir / "calls.parquet")
    calls["run_id"] = cfg.run_id
    calls.to_parquet(out_dir / "calls.parquet", index=False)
    episodes = [
        {
            "episode_id": int(e),
            "task_id": int(rows["task_id"].iloc[0]),
            "n_calls": int(len(rows)),
            "success": bool(int(e) % 2),
            "censored": False,
            "is_probe_episode": True,
        }
        for e, rows in calls.groupby("episode_id")
    ]
    manifest = {
        "n_calls": int(len(calls)),
        "n_episodes": len(episodes),
        "episodes": episodes,
        "stop_reason": "completed",
        "verification": {"passed": True, "max_abs_logprob_diff": 4.8e-7},
        "config_hash": cfg.hash(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
