"""Thin wrapper around RLinf's Wan action-conditioned world model.

Calls the same ``WanVideoPipeline`` that ``rlinf/envs/world_model/
world_model_wan_env.py`` builds, with the same arguments, but statelessly:
``WanEnv`` keeps a rolling frame queue and can only reset from its own
dataset, whereas the comparator needs arbitrary (context, chunk) pairs taken
from real LIBERO rollouts.  Mirrored from ``WanEnv`` on purpose:

* only the DiT and the VAE are loaded (no text encoder: the pipeline is never
  given a prompt);
* the action tensor is float32 on the device, ``[B, 13, 7]`` = 5 condition
  slots + the 8-step chunk; generation runs under bf16 autocast;
* 256 x 256, 13 frames, ``cfg_scale=1.0``, frames 5..12 are the prediction.

One deliberate difference: every candidate chunk is generated **one at a
time at the same seed**.  The pipeline draws noise for the whole batch from
one generator, so a sample's noise depends on its batch index; batching
candidates would mix "different action" with "different noise", which is the
exact distinction E3 measures.

Requires a GPU and the ``wan`` install (``bash requirements/install.sh
embodied --model openvla-oft --env wan``), which provides ``diffsynth`` from
``RLinf/diffsynth-studio``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np

N_CONDITION = 5
N_FRAMES = 13
FRAME_SIZE = 256


class WanWorldModel:
    """Predict the 8 frames an action chunk produces from a given context."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        device: str = "cuda",
        num_inference_steps: int = 5,
        model_file: str = "model-00001.safetensors",
        vae_file: str = "Wan2.2_VAE.pth",
    ) -> None:
        import torch
        from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

        checkpoint_dir = Path(checkpoint_dir)
        for name in (model_file, vae_file):
            if not (checkpoint_dir / name).is_file():
                raise FileNotFoundError(
                    f"{checkpoint_dir / name} missing; download RLinf/RLinf-Wan-LIBERO-Goal"
                )
        self._torch = torch
        self.device = torch.device(device)
        self.num_inference_steps = int(num_inference_steps)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=str(self.device),
            model_configs=[
                ModelConfig(
                    path=str(checkpoint_dir / model_file), offload_device="cpu"
                ),
                ModelConfig(path=str(checkpoint_dir / vae_file), offload_device="cpu"),
            ],
        )
        self.pipe.dit.to(self.device)
        self.pipe.vae.to(self.device)

    @staticmethod
    def _pil(frame: np.ndarray):
        from PIL import Image

        frame = np.asarray(frame, dtype=np.uint8)
        if frame.shape != (FRAME_SIZE, FRAME_SIZE, 3):
            raise ValueError(
                f"WM frames must be {FRAME_SIZE}x{FRAME_SIZE}x3 uint8 "
                f"(the pipeline does not resize in batched mode); got {frame.shape}"
            )
        return Image.fromarray(frame)

    def predict(
        self,
        reference: np.ndarray,
        context: np.ndarray,
        condition: np.ndarray,
        chunk: np.ndarray,
        seed: int = 0,
    ) -> np.ndarray:
        """One chunk -> ``[8, 256, 256, 3]`` uint8 predicted frames."""
        torch = self._torch
        chunk = np.asarray(chunk, dtype=np.float32)
        condition = np.asarray(condition, dtype=np.float32)
        if condition.shape != (N_CONDITION, chunk.shape[-1]):
            raise ValueError(f"condition must be [5, 7], got {condition.shape}")
        if len(context) != 4:
            raise ValueError(f"context must hold 4 frames, got {len(context)}")
        actions = np.concatenate([condition, chunk], axis=0)[None]  # [1, 13, 7]
        if actions.shape[1] != N_FRAMES:
            raise ValueError(f"expected 5 + 8 action rows, got {actions.shape[1]}")
        action_tensor = torch.from_numpy(actions).to(self.device)
        kwargs = {
            "seed": int(seed),
            "tiled": False,
            "input_image": [self._pil(reference)],
            "input_image4": [[self._pil(frame) for frame in context]],
            "action": action_tensor,
            "height": FRAME_SIZE,
            "width": FRAME_SIZE,
            "num_frames": N_FRAMES,
            "num_inference_steps": self.num_inference_steps,
            "cfg_scale": 1.0,
            "progress_bar_cmd": lambda x: x,
            "batch_size": 1,
        }
        with (
            torch.no_grad(),
            torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16),
        ):
            output = self.pipe(**kwargs)
        frames = [
            np.asarray(image.convert("RGB"), dtype=np.uint8) for image in output[0]
        ]
        return np.stack(frames[N_CONDITION:])

    def predict_candidates(
        self,
        reference: np.ndarray,
        context: np.ndarray,
        condition: np.ndarray,
        candidates: Mapping[str, np.ndarray],
        seed: int = 0,
        extra_seed: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        """Every candidate at the same seed; ``cmd`` also at ``extra_seed``."""
        predictions = {
            name: self.predict(reference, context, condition, chunk, seed)
            for name, chunk in candidates.items()
        }
        if extra_seed is not None and "cmd" in candidates:
            predictions["cmd@noise"] = self.predict(
                reference, context, condition, candidates["cmd"], extra_seed
            )
        return predictions


class ToyWorldModel:
    """Numpy stand-in with the same interface, for tests and dry runs only.

    "Predicts" by shifting the last context frame along x/y by the chunk's
    summed translation (``pixels_per_unit`` per unit of action), frame by
    frame, plus seed-dependent noise.  It is action-sensitive by construction,
    or action-blind with ``blind=True``; neither says anything about Wan.
    """

    def __init__(
        self, pixels_per_unit: float = 6.0, noise: float = 2.0, blind: bool = False
    ) -> None:
        self.pixels_per_unit = pixels_per_unit
        self.noise = noise
        self.blind = blind

    def predict(
        self,
        reference: np.ndarray,
        context: np.ndarray,
        condition: np.ndarray,
        chunk: np.ndarray,
        seed: int = 0,
    ) -> np.ndarray:
        base = np.asarray(context[-1], dtype=np.float32)
        rng = np.random.default_rng(seed)
        chunk = np.zeros_like(chunk) if self.blind else np.asarray(chunk, np.float32)
        offsets = np.cumsum(chunk[:, :2], axis=0) * self.pixels_per_unit
        frames = []
        for dx, dy in offsets:
            shifted = np.roll(base, (int(round(dy)), int(round(dx))), axis=(0, 1))
            shifted = shifted + rng.normal(0.0, self.noise, size=shifted.shape)
            frames.append(np.clip(shifted, 0, 255).astype(np.uint8))
        return np.stack(frames)

    def predict_candidates(
        self,
        reference: np.ndarray,
        context: np.ndarray,
        condition: np.ndarray,
        candidates: Mapping[str, np.ndarray],
        seed: int = 0,
        extra_seed: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        predictions = {
            name: self.predict(reference, context, condition, chunk, seed)
            for name, chunk in candidates.items()
        }
        if extra_seed is not None and "cmd" in candidates:
            predictions["cmd@noise"] = self.predict(
                reference, context, condition, candidates["cmd"], extra_seed
            )
        return predictions


__all__ = ["ToyWorldModel", "WanWorldModel"]
