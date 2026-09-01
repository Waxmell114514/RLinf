"""Generate a synthetic run directory with known ground truth.

Lets the analysis half be exercised end-to-end without a GPU, LIBERO, or a
checkpoint.  The generator plants a *strong* mechanical signal (the executed
chunk really does drive the state delta, so P2 must be near-perfect) and a
*tunable* hidden-state signal, so the probe code can be checked against
outcomes we already know.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efference_probe.hijack import (  # noqa: E402
    HIJACK,
    SELF,
    HijackScheduler,
    build_executed_chunk,
)

N_CHUNKS = 8
ACTION_DIM = 7
N_ACT = N_CHUNKS * ACTION_DIM
STATE_DIM = 8
LAYERS = [0, 4, 8]
POOLS = ["ctx_mean", "ctx_last", "act_mean"]


def make_synthetic_run(
    out_dir: str | Path,
    n_tasks: int = 3,
    episodes_per_task: int = 8,
    n_calls: int = 20,
    hidden_size: int = 64,
    hidden_signal: float = 0.8,
    p_hijack: float = 0.25,
    transform: str = "swap",
    seed: int = 0,
    success_probability: float = 0.3,
    command_mean: float = 0.0,
    command_autocorr: float = 0.0,
) -> Path:
    """Write a run directory that looks like a real collection.

    Args:
        hidden_signal: how strongly the post-transition hidden state encodes
            the hijack.  0 makes P1 a null; large values make it decodable.
        command_mean: per-episode drift added to every commanded delta.  Zero
            gives i.i.d. zero-mean commands; a nonzero value imitates directed
            reaching, where a mean-shifting transform is separable from the
            marginal state delta alone.
        command_autocorr: AR(1) coefficient linking each command to the
            previous one.  The policy is open-loop either way -- it never sees
            the state -- so any "undo alignment" measured here is an artefact.
    """
    out_dir = Path(out_dir)
    (out_dir / "hidden").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    rows = []
    episode_id = 0
    for task_id in range(n_tasks):
        num_envs = episodes_per_task
        scheduler = HijackScheduler(
            num_envs=num_envs,
            p_hijack=p_hijack,
            warmup_calls=4,
            min_clean_gap=2,
            transform=transform,
            rng=np.random.default_rng([seed, task_id]),
        )
        episode_ids = np.arange(episode_id, episode_id + num_envs)
        episode_id += num_envs
        recording = np.ones(num_envs, dtype=bool)
        first_success = {int(e): -1 for e in episode_ids}
        states = rng.normal(size=(num_envs, STATE_DIM)).astype(np.float32)
        hidden_buffer: dict[int, list[np.ndarray]] = {int(e): [] for e in episode_ids}
        hidden_calls: dict[int, list[int]] = {int(e): [] for e in episode_ids}
        episode_rows: dict[int, list[dict]] = {int(e): [] for e in episode_ids}
        was_hijacked = np.zeros(num_envs, dtype=bool)
        previous_command = None

        for call_idx in range(n_calls):
            if not recording.any():
                break
            noise = rng.normal(scale=0.3, size=(num_envs, N_CHUNKS, ACTION_DIM)).astype(
                np.float32
            )
            if command_autocorr and previous_command is not None:
                commanded = command_autocorr * previous_command + noise
            else:
                commanded = noise
            previous_command = commanded.copy()
            commanded[..., :6] += command_mean
            decisions = scheduler.decide(call_idx, recording)
            executed = build_executed_chunk(commanded, decisions)

            states_before = states.copy()
            # The state really is driven by the executed chunk, so
            # a_cmd + dstates is a near-perfect mechanical oracle.
            displacement = executed.sum(axis=1)
            states_after = states_before.copy()
            states_after[:, :ACTION_DIM] += displacement
            states_after += rng.normal(scale=0.01, size=states_after.shape)

            for slot in range(num_envs):
                if not recording[slot]:
                    continue
                episode = int(episode_ids[slot])
                decision = decisions[slot]
                hidden = rng.normal(size=(len(LAYERS), len(POOLS), hidden_size)).astype(
                    np.float32
                )
                # Deeper layers carry more of the planted signal, mimicking the
                # depth profile a real probe would sweep for.
                if was_hijacked[slot]:
                    for layer_index in range(len(LAYERS)):
                        weight = hidden_signal * (layer_index / max(len(LAYERS) - 1, 1))
                        hidden[layer_index, :, :4] += weight
                hidden_buffer[episode].append(hidden.astype(np.float16))
                hidden_calls[episode].append(call_idx)

                episode_rows[episode].append(
                    {
                        "run_id": "synthetic",
                        "episode_id": episode,
                        "task_id": task_id,
                        "reset_state_id": episode,
                        "env_slot": slot,
                        "call_idx": call_idx,
                        "label": HIJACK if decision.hijacked else SELF,
                        "transform": decision.transform or "",
                        "donor_env_slot": (
                            decision.donor_env_index
                            if decision.donor_env_index is not None
                            else -1
                        ),
                        "donor_episode_id": (
                            int(episode_ids[decision.donor_env_index])
                            if decision.donor_env_index is not None
                            else -1
                        ),
                        "skipped_reason": decision.skipped_reason or "",
                        "a_cmd_model": commanded[slot].reshape(-1),
                        "a_cmd_env": commanded[slot].reshape(-1),
                        "a_exec": executed[slot].reshape(-1),
                        "states_before": states_before[slot],
                        "states_after": states_after[slot],
                        "logprob_sum": float(rng.normal(-20, 2)),
                        "logprobs": rng.normal(-0.4, 0.1, size=N_ACT).astype(
                            np.float32
                        ),
                        "entropy_mean": float(
                            rng.normal(1.2 + 0.2 * was_hijacked[slot], 0.2)
                        ),
                        "entropy_tokens": rng.normal(1.2, 0.2, size=N_ACT).astype(
                            np.float32
                        ),
                        "reward": 0.0,
                        "success_substep": -1,
                        "terminated": False,
                        "truncated": False,
                    }
                )

            was_hijacked = np.array([d.hijacked for d in decisions])
            states = states_after
            # Retire a few episodes so success handling is exercised too.
            if call_idx > 10:
                finishing = (
                    rng.random(num_envs) < success_probability / max(n_calls - 10, 1)
                ) & recording
                for slot in np.flatnonzero(finishing):
                    first_success[int(episode_ids[slot])] = call_idx
                recording &= ~finishing

        for episode in episode_ids:
            episode = int(episode)
            success_call = first_success[episode]
            for row in episode_rows[episode]:
                row["first_success_call"] = success_call
                row["success_flag"] = bool(
                    success_call >= 0 and row["call_idx"] >= success_call
                )
                row["post_success"] = bool(
                    success_call >= 0 and row["call_idx"] > success_call
                )
                rows.append(row)
            if hidden_buffer[episode]:
                np.savez_compressed(
                    out_dir / "hidden" / f"ep{episode:05d}.npz",
                    h=np.stack(hidden_buffer[episode]),
                    call_idx=np.asarray(hidden_calls[episode], dtype=np.int32),
                )

    frame = pd.DataFrame(rows)
    frame.to_parquet(out_dir / "calls.parquet", index=False)

    config = {
        "run_id": "synthetic",
        "config_hash": "synthetic",
        "git_sha": "synthetic",
        "config": {
            "hijack": {
                "min_clean_gap": 2,
                "warmup_calls": 4,
                "transform": transform,
                "p_hijack": p_hijack,
                "probe_episode_fraction": 1.0,
            },
            "model": {"num_action_chunks": N_CHUNKS, "action_dim": ACTION_DIM},
            "env": {"task_suite_name": "synthetic"},
        },
        "schema": {
            "array_columns": {},
            "num_action_chunks": N_CHUNKS,
            "action_dim": ACTION_DIM,
            "n_act": N_ACT,
            "state_dim": STATE_DIM,
            "layers": LAYERS,
            "pools": POOLS,
        },
    }
    with open(out_dir / "run_config.yaml", "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return out_dir


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir")
    parser.add_argument("--hidden-signal", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    path = make_synthetic_run(
        args.out_dir, hidden_signal=args.hidden_signal, seed=args.seed
    )
    print(f"wrote synthetic run to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
