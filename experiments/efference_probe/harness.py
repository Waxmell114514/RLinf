"""Single-process collection loop: LIBERO + OpenVLA-OFT with hijacked chunks.

Deliberately bypasses Ray and the worker abstractions.  We only need one GPU
and one env vector, and the worker layer exists to move tensors between
processes -- machinery that would buy us nothing here and cost debugging time.
What we *do* reuse verbatim are the pieces where drift would be silent:
``rlinf.models.get_model`` for checkpoint loading, ``LiberoEnv`` for the sim,
and ``rlinf.envs.action_utils.prepare_actions`` for the gripper convention.

Per model call ``m`` the loop is::

    obs -> predict_action_batch  (hook captures h_m)
        -> prepare_actions       (model space -> env space)
        -> hijack schedule       (SELF or a transform of the commanded chunk)
        -> chunk_step            (8 sim steps)
        -> obs'                  (input to call m+1, whose h_{m+1} is the probe target)

Env config choices worth knowing about:

* ``auto_reset: False`` and ``ignore_terminations: False`` -- episodes are
  driven explicitly, so ``chunk_step`` reports real per-substep terminations
  and the first-success call can be pinned exactly.
* ``is_eval: True`` -- LIBERO envs are rebuilt only when the *task* changes.
  Because one batch holds many initial states of a single task, that is one
  rebuild per task instead of one per episode.
* One task per batch also makes every swap donor a same-task episode, which is
  what makes T3 a control for "weird actions" rather than a new confound.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from .capture import HiddenStateCapture
from .config import RunConfig
from .hijack import HIJACK, SELF, HijackScheduler, build_executed_chunk
from .storage import ARRAY_COLUMNS, DiskBudget, RunWriter, flatten, git_sha

# Repo root, i.e. the directory holding `rlinf/` and `examples/`.
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_DIR = REPO_ROOT / "examples" / "embodiment" / "config" / "model"
ENV_CONFIG_DIR = REPO_ROOT / "examples" / "embodiment" / "config" / "env"


def build_model_config(cfg: RunConfig) -> DictConfig:
    """Start from the repo's own model yaml, then apply run overrides.

    Loading the shipped yaml rather than hand-writing the fields means a future
    change to ``model/openvla_oft.yaml`` reaches this harness too.
    """
    base_path = MODEL_CONFIG_DIR / f"{cfg.model.model_type}.yaml"
    if not base_path.is_file():
        raise FileNotFoundError(f"no model config at {base_path}")
    model_cfg = OmegaConf.load(base_path)
    with open_dict(model_cfg):
        model_cfg.model_path = cfg.model.model_path
        model_cfg.implement_version = cfg.model.implement_version
        model_cfg.unnorm_key = cfg.model.unnorm_key
        model_cfg.precision = cfg.model.precision
        model_cfg.action_dim = cfg.model.action_dim
        model_cfg.num_action_chunks = cfg.model.num_action_chunks
        model_cfg.max_prompt_length = cfg.model.max_prompt_length
        model_cfg.num_images_in_input = cfg.model.num_images_in_input
        model_cfg.center_crop = cfg.model.center_crop
        model_cfg.attn_implementation = cfg.model.attn_implementation
        # Inference only: no value head, no LoRA, no training-side extras.
        model_cfg.add_value_head = False
        model_cfg.is_lora = False
        model_cfg.lora_path = None
    return model_cfg


def build_env_config(cfg: RunConfig) -> DictConfig:
    """Start from the repo's env yaml for the suite, then apply run overrides."""
    base_path = ENV_CONFIG_DIR / f"{cfg.env.task_suite_name}.yaml"
    if not base_path.is_file():
        raise FileNotFoundError(f"no env config at {base_path}")
    env_cfg = OmegaConf.load(base_path)
    max_steps = cfg.env.max_calls_per_episode * cfg.model.num_action_chunks
    with open_dict(env_cfg):
        env_cfg.task_suite_name = cfg.env.task_suite_name
        env_cfg.total_num_envs = cfg.env.num_envs
        env_cfg.seed = cfg.env.seed
        env_cfg.group_size = 1
        # Episode boundaries are ours to manage; see the module docstring.
        env_cfg.auto_reset = False
        env_cfg.ignore_terminations = False
        env_cfg.is_eval = True
        env_cfg.use_fixed_reset_state_ids = True
        env_cfg.use_ordered_reset_state_ids = True
        env_cfg.max_episode_steps = max_steps
        env_cfg.max_steps_per_rollout_epoch = max_steps
        env_cfg.reset_gripper_open = cfg.env.reset_gripper_open
        env_cfg.skip_intermediate_renders = cfg.env.skip_intermediate_renders
        env_cfg.video_cfg = OmegaConf.create(
            {"save_video": False, "info_on_video": False, "video_base_dir": ""}
        )
        env_cfg.init_params = OmegaConf.create(
            {
                "camera_heights": cfg.env.camera_size,
                "camera_widths": cfg.env.camera_size,
            }
        )
    return env_cfg


class EfferenceHarness:
    """Collects hijack-labelled calls with per-call hidden states."""

    def __init__(self, cfg: RunConfig, logbook: Optional["Logbook"] = None) -> None:
        cfg.validate()
        self.cfg = cfg
        self.logbook = logbook
        self.out_dir = Path(cfg.out_root) / cfg.run_id
        self.budget = DiskBudget(
            self.out_dir, cfg.budget.max_disk_gb, cfg.budget.max_gpu_hours
        )
        self.writer = RunWriter(self.out_dir, self.budget)
        self.model_cfg = build_model_config(cfg)
        self.env_cfg = build_env_config(cfg)

        self.model = None
        self.env = None
        self.capture: Optional[HiddenStateCapture] = None
        self._episode_counter = 0
        self._last_heartbeat = time.time()
        self._verification: dict[str, Any] = {}
        self._stop_reason: Optional[str] = None

    # -- setup -----------------------------------------------------------

    def setup(self) -> None:
        from rlinf.envs.libero.libero_env import LiberoEnv
        from rlinf.models import get_model

        self.log(f"loading checkpoint {self.cfg.model.model_path}")
        self.model = get_model(self.model_cfg)
        if self.model is None:
            raise RuntimeError(
                f"no model builder registered for {self.cfg.model.model_type!r}"
            )
        self.model.eval()

        self.capture = HiddenStateCapture(
            self.model,
            layers=self.cfg.capture.layers,
            pools=self.cfg.capture.pools,
            capture_vision=self.cfg.capture.capture_vision,
        )
        if self.cfg.capture.capture_vision and not self.capture.vision_available:
            self.log(f"WARNING: {self.capture.vision_warning}")

        self.log(f"building LiberoEnv ({self.cfg.env.num_envs} parallel envs)")
        self.env = LiberoEnv(
            cfg=self.env_cfg,
            num_envs=self.cfg.env.num_envs,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )
        # Take over reset-state selection: LiberoEnv.reset() otherwise ignores
        # the ids we pass on its very first call.
        self.env.is_start = False

    def teardown(self) -> None:
        if self.capture is not None:
            self.capture.close()
        # LiberoEnv holds a pool of subprocesses; leaving them behind wedges
        # the GPU for the next run.
        inner = getattr(self.env, "env", None)
        if inner is not None and hasattr(inner, "close"):
            try:
                inner.close()
            except Exception as error:  # noqa: BLE001 - teardown is best effort
                self.log(f"WARNING: env close failed: {type(error).__name__}: {error}")

    # -- logging ---------------------------------------------------------

    def log(self, message: str) -> None:
        stamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(stamped, flush=True)
        if self.logbook is not None:
            self.logbook.write(message)

    def heartbeat(self, message: str) -> None:
        now = time.time()
        if now - self._last_heartbeat >= self.cfg.budget.heartbeat_seconds:
            self._last_heartbeat = now
            usage = self.budget.summary()
            self.log(
                f"heartbeat: {message} | "
                f"{usage['elapsed_hours']:.2f}/{usage['max_hours']:.1f} h, "
                f"{usage['disk_gb']:.2f}/{usage['max_disk_gb']:.0f} GB, "
                f"{self.writer.n_calls} calls"
            )

    # -- task / reset bookkeeping ----------------------------------------

    def task_ids(self) -> list[int]:
        if self.cfg.env.task_ids is not None:
            return list(self.cfg.env.task_ids)
        return list(range(self.env.task_suite.get_num_tasks()))

    def reset_state_ids_for_task(self, task_id: int) -> np.ndarray:
        """Global reset-state ids for the first N trials of ``task_id``.

        LIBERO indexes initial states globally across (task, trial); the
        cumulative bin edges in ``LiberoEnv`` give the offset for each task.
        """
        bins = self.env.cumsum_trial_id_bins
        start = int(bins[task_id - 1]) if task_id > 0 else 0
        available = int(bins[task_id]) - start
        count = min(self.cfg.env.init_states_per_task, available)
        if count < self.cfg.env.init_states_per_task:
            self.log(
                f"task {task_id} has only {available} initial states; using {count}"
            )
        return start + np.arange(count, dtype=np.int64)

    # -- main loop -------------------------------------------------------

    def run(self) -> Path:
        self.setup()
        self.writer.write_run_config(self._run_config_payload())
        try:
            for task_id in self.task_ids():
                reset_ids = self.reset_state_ids_for_task(task_id)
                num_envs = self.cfg.env.num_envs
                for batch_index in range(0, len(reset_ids), num_envs):
                    batch_ids = reset_ids[batch_index : batch_index + num_envs]
                    self._run_batch(task_id, batch_ids, batch_index // num_envs)
                    exceeded = self.budget.exceeded()
                    if exceeded is not None:
                        self._stop_reason = f"{exceeded}_budget_exhausted"
                        self.log(f"STOP: {self._stop_reason}")
                        break
                if self._stop_reason is not None:
                    break
        finally:
            self.teardown()

        parquet_path = self.writer.finalize(
            {
                "stop_reason": self._stop_reason or "completed",
                "verification": self._verification,
                "config_hash": self.cfg.hash(),
            }
        )
        self.log(
            f"done: {self.writer.n_calls} calls -> {parquet_path} "
            f"({self.budget.summary()['disk_gb']:.2f} GB)"
        )
        return parquet_path

    def _run_batch(self, task_id: int, batch_ids: np.ndarray, batch_index: int) -> None:
        """Run one batch: ``len(batch_ids)`` episodes of a single task."""
        num_envs = self.cfg.env.num_envs
        n_real = len(batch_ids)
        # LiberoEnv has a fixed env count; pad short batches and never record
        # the padded slots.
        if n_real < num_envs:
            padding = np.repeat(batch_ids[-1], num_envs - n_real)
            padded_ids = np.concatenate([batch_ids, padding])
        else:
            padded_ids = batch_ids

        env_indices = np.arange(num_envs)
        obs, _ = self.env.reset(env_idx=env_indices, reset_state_ids=padded_ids)

        episode_ids = np.full(num_envs, -1, dtype=np.int64)
        for slot in range(n_real):
            episode_ids[slot] = self._episode_counter
            self._episode_counter += 1

        recording = np.zeros(num_envs, dtype=bool)
        recording[:n_real] = True

        # Hijack RNG is derived from (seed, task, batch) so a rerun of a single
        # batch reproduces its schedule without replaying the whole suite.
        rng = np.random.default_rng([self.cfg.env.seed, task_id, batch_index, 0xEFFE])
        # Some episodes run completely clean so E3 can compare success rates
        # within the run instead of paying for a second collection pass.
        is_probe_episode = np.zeros(num_envs, dtype=bool)
        if self.cfg.hijack.enabled:
            is_probe_episode[:n_real] = (
                rng.random(n_real) < self.cfg.hijack.probe_episode_fraction
            )
        scheduler = HijackScheduler(
            num_envs=num_envs,
            p_hijack=self.cfg.hijack.p_hijack if self.cfg.hijack.enabled else 0.0,
            warmup_calls=self.cfg.hijack.warmup_calls,
            min_clean_gap=self.cfg.hijack.min_clean_gap,
            transform=self.cfg.hijack.transform,
            rng=rng,
        )

        rows: dict[int, list[dict[str, Any]]] = {
            int(episode_ids[slot]): [] for slot in range(n_real)
        }
        first_success_call = {int(episode_ids[slot]): -1 for slot in range(n_real)}
        n_hijacks = {int(episode_ids[slot]): 0 for slot in range(n_real)}

        sampling = self._sampling_params()
        for call_idx in range(self.cfg.env.max_calls_per_episode):
            if not recording.any():
                break

            commanded, result, captured = self._policy_step(obs, sampling, call_idx)
            commanded_env = self._to_env_action_space(commanded)
            # Clean episodes are never hijacked, but stay eligible as swap
            # donors: their commanded chunks are ordinary policy output.
            decisions = scheduler.decide(
                call_idx,
                active=recording & is_probe_episode,
                donor_active=recording,
            )
            executed = build_executed_chunk(commanded_env, decisions)

            states_before = _as_numpy(obs["states"])
            main_images = _as_numpy(obs["main_images"])
            wrist_images = _as_numpy(obs.get("wrist_images"))

            next_obs, rewards, terminations, truncations, _infos = self.env.chunk_step(
                executed
            )
            obs_after = (
                next_obs[-1] if isinstance(next_obs, (list, tuple)) else next_obs
            )
            states_after = _as_numpy(obs_after["states"])

            terminated = terminations.cpu().numpy().astype(bool)
            truncated = truncations.cpu().numpy().astype(bool)
            reward_sum = rewards.cpu().numpy().astype(np.float32).sum(axis=1)

            for slot in range(num_envs):
                if not recording[slot]:
                    continue
                episode_id = int(episode_ids[slot])
                decision = decisions[slot]
                success_substep = (
                    int(np.argmax(terminated[slot])) if terminated[slot].any() else -1
                )
                row = {
                    "run_id": self.cfg.run_id,
                    "episode_id": episode_id,
                    "task_id": int(task_id),
                    "reset_state_id": int(padded_ids[slot]),
                    "env_slot": int(slot),
                    "call_idx": int(call_idx),
                    "is_probe_episode": bool(is_probe_episode[slot]),
                    "label": HIJACK if decision.hijacked else SELF,
                    "transform": decision.transform or "",
                    "donor_env_slot": (
                        int(decision.donor_env_index)
                        if decision.donor_env_index is not None
                        else -1
                    ),
                    "donor_episode_id": (
                        int(episode_ids[decision.donor_env_index])
                        if decision.donor_env_index is not None
                        else -1
                    ),
                    "skipped_reason": decision.skipped_reason or "",
                    "a_cmd_model": flatten(commanded[slot]),
                    "a_cmd_env": flatten(commanded_env[slot]),
                    "a_exec": flatten(executed[slot]),
                    "states_before": flatten(states_before[slot]),
                    "states_after": flatten(states_after[slot]),
                    "logprob_sum": float(result["prev_logprobs"][slot].sum().item()),
                    "logprobs": flatten(
                        result["prev_logprobs"][slot].float().cpu().numpy()
                    ),
                    "entropy_mean": float(captured.entropy_mean[slot]),
                    "entropy_tokens": flatten(captured.entropy_tokens[slot]),
                    "reward": float(reward_sum[slot]),
                    "success_substep": success_substep,
                    "terminated": bool(terminated[slot].any()),
                    "truncated": bool(truncated[slot].any()),
                }
                rows[episode_id].append(row)
                if decision.hijacked:
                    n_hijacks[episode_id] += 1

                self.writer.add_hidden(
                    episode_id,
                    call_idx,
                    captured.hidden[slot],
                    captured.vision[slot] if captured.vision is not None else None,
                )
                if self._should_save_frames(episode_id):
                    self.writer.save_frames(
                        episode_id,
                        call_idx,
                        main_images[slot],
                        (
                            wrist_images[slot]
                            if wrist_images is not None and self.cfg.frames.save_wrist
                            else None
                        ),
                        quality=self.cfg.frames.jpeg_quality,
                    )

                if terminated[slot].any():
                    first_success_call[episode_id] = call_idx
                    recording[slot] = False
                elif truncated[slot].any():
                    recording[slot] = False

            obs = obs_after
            self.heartbeat(f"task {task_id} batch {batch_index} call {call_idx}")
            if self.budget.exceeded() is not None:
                break

        self._finish_batch(rows, first_success_call, n_hijacks, task_id)

    def _finish_batch(
        self,
        rows: dict[int, list[dict[str, Any]]],
        first_success_call: dict[int, int],
        n_hijacks: dict[int, int],
        task_id: int,
    ) -> None:
        """Stamp episode-level outcomes onto rows and flush to disk."""
        for episode_id, episode_rows in rows.items():
            success_call = first_success_call[episode_id]
            for row in episode_rows:
                row["first_success_call"] = success_call
                row["success_flag"] = bool(
                    success_call >= 0 and row["call_idx"] >= success_call
                )
                # Calls after first success carry post-hoc-contaminated states;
                # SPEC 2 excludes them from probe data.
                row["post_success"] = bool(
                    success_call >= 0 and row["call_idx"] > success_call
                )
                self.writer.add_call(row)
            self.writer.add_episode(
                {
                    "episode_id": episode_id,
                    "task_id": int(task_id),
                    "n_calls": len(episode_rows),
                    "first_success_call": success_call,
                    "success": bool(success_call >= 0),
                    "n_hijacks": n_hijacks[episode_id],
                    "is_probe_episode": bool(
                        episode_rows[0]["is_probe_episode"] if episode_rows else False
                    ),
                }
            )
            self.writer.flush_episode(episode_id)

    # -- policy ----------------------------------------------------------

    def _sampling_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"do_sample": self.cfg.model.do_sample}
        if self.cfg.model.do_sample:
            params["temperature"] = self.cfg.model.temperature
            params["top_k"] = self.cfg.model.top_k
            params["top_p"] = self.cfg.model.top_p
        return params

    def _policy_step(
        self, obs: dict[str, Any], sampling: dict[str, Any], call_idx: int
    ):
        """One ``predict_action_batch`` with hidden-state capture armed."""
        # predict_action_batch unsqueezes `main_images` in place; hand it a
        # shallow copy so our logged observation keeps its [B, H, W, C] shape.
        model_obs = dict(obs)
        verify = self.cfg.capture.verify_indexing and not self._verification
        if verify:
            self.capture.arm_verification()
        with torch.no_grad(), self.capture.armed():
            actions, result = self.model.predict_action_batch(
                env_obs=model_obs, **sampling
            )
        captured = self.capture.take()
        if verify:
            self._verification = self.capture.verify_indexing(
                action_tokens=result["forward_inputs"]["action_tokens"],
                model_logprobs=result["prev_logprobs"],
                do_sample=self.cfg.model.do_sample,
                temperature=self.cfg.model.temperature,
                top_k=self.cfg.model.top_k,
            )
            self._verification.update(captured.meta)
            self.log(f"indexing check passed: {self._verification}")
        commanded = _as_numpy(actions).astype(np.float32)
        return commanded, result, captured

    def _to_env_action_space(self, commanded: np.ndarray) -> np.ndarray:
        """Apply the repo's model->env action mapping to a *copy*.

        ``prepare_actions_for_libero`` rewrites the gripper channel in place and
        does not copy its input, so passing the policy's own array would
        silently corrupt the commanded actions we are trying to log.
        """
        from rlinf.envs.action_utils import prepare_actions

        return np.asarray(
            prepare_actions(
                raw_chunk_actions=commanded.copy(),
                env_type=self.env_cfg.env_type,
                model_type=self.cfg.model.model_type,
                num_action_chunks=self.cfg.model.num_action_chunks,
                action_dim=self.cfg.model.action_dim,
                env_cfg=self.env_cfg,
            ),
            dtype=np.float32,
        )

    # -- misc ------------------------------------------------------------

    def _should_save_frames(self, episode_id: int) -> bool:
        if self.cfg.frames.mode == "none":
            return False
        if self.cfg.frames.mode == "all":
            return True
        return episode_id < self.cfg.frames.max_episodes

    def _run_config_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.cfg.run_id,
            "config_hash": self.cfg.hash(),
            "git_sha": git_sha(REPO_ROOT),
            "config": self.cfg.to_dict(),
            "resolved_model_config": OmegaConf.to_container(
                self.model_cfg, resolve=True
            ),
            "resolved_env_config": OmegaConf.to_container(self.env_cfg, resolve=True),
            "schema": {
                "array_columns": {k: list(v) for k, v in ARRAY_COLUMNS.items()},
                "num_action_chunks": self.cfg.model.num_action_chunks,
                "action_dim": self.cfg.model.action_dim,
                "n_act": self.cfg.model.num_action_chunks * self.cfg.model.action_dim,
                "state_dim": 8,
                "layers": list(self.cfg.capture.layers),
                "pools": list(self.cfg.capture.pools),
            },
        }


def _as_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class Logbook:
    """Append-only research logbook (SPEC 0.6)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, "a") as handle:
            handle.write(f"- `{stamp}` {message}\n")


__all__ = [
    "EfferenceHarness",
    "Logbook",
    "build_env_config",
    "build_model_config",
]
