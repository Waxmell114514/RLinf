"""Hidden-state capture for OpenVLA-OFT, without touching ``rlinf/``.

``OpenVLAOFTForRLActionPrediction.predict_action_batch`` already runs its
language-model forward with ``output_hidden_states=True``, so every layer's
activations are computed whether or not anyone reads them.  We attach a
``forward`` hook to ``model.language_model``, pool the tensors we want, and
move them to CPU.  No second forward pass, no monkeypatching, no fork of the
model class.

Sequence layout (derived from ``predict_action_batch`` and asserted at
runtime).  With ``n_act = action_dim * num_action_chunks = 56``::

    [ BOS | vision patches | rest of prompt | 56 action placeholders ]
      0     1 .. n_patches   .. seq-n_act-1   seq-n_act .. seq-1

``predict_action_batch`` reads its action logits from
``logits[:, n_patches + n_prompt_tokens : + n_act]`` and its action hidden
states from ``hidden[:, -n_act-1 : -1]``.  Those two slices are the same
window, because ``seq = n_patches + n_prompt + 1 + n_act``.  So the positions
that actually drive action decoding are ``[-n_act-1 : -1]`` -- one *before*
the placeholder block, in the usual next-token-prediction offset.

That is a correction to SPEC 2, which described the pools as ``[-56:]`` /
``[:-56]`` / ``-57``.  The placeholder block really is the last 56 positions,
but its hidden states are not what the model reads: the readout window is
shifted one step left.  Pools are therefore defined as:

``act_mean``
    mean over ``[-n_act-1 : -1]`` -- exactly the window the model decodes from.
``ctx_last``
    position ``-n_act-1`` -- the last real prompt token (the trailing space).
    This is also the position the value head reads, and the first element of
    the readout window: the model's summary of the scene immediately before it
    commits to an action.
``ctx_mean``
    attention-masked mean over ``[: -n_act-1]`` -- BOS, vision patches and the
    prompt.  The mask matters because prompts are left-padded to
    ``max_prompt_length``; an unmasked mean would average in pad positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from .config import POOL_NAMES


@dataclass
class CapturedCall:
    """Everything the hook pulled out of a single model call."""

    # [num_envs, n_layers_kept, n_pools, hidden_size] float16 on CPU.
    hidden: np.ndarray
    # [num_envs] float32: mean per-token entropy of the action distribution.
    entropy_mean: np.ndarray
    # [num_envs, n_act] float32: per action-token entropy.
    entropy_tokens: np.ndarray
    # [num_envs, d_proj] float16, or None when vision capture is off.
    vision: Optional[np.ndarray] = None
    # Diagnostics recorded once per run.
    meta: dict[str, Any] = field(default_factory=dict)


class HiddenStateCapture:
    """Forward-hook based capture of LM hidden states and action logits.

    Usage::

        capture = HiddenStateCapture(model, layers=[0, 16, 32], pools=[...])
        with capture.armed():
            actions, result = model.predict_action_batch(env_obs=obs, ...)
        captured = capture.take()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layers: list[int],
        pools: list[str],
        capture_vision: bool = False,
    ) -> None:
        self.model = model
        self.layers = list(layers)
        self.pools = list(pools)
        for pool in self.pools:
            if pool not in POOL_NAMES:
                raise ValueError(f"unknown pool {pool!r}; expected {POOL_NAMES}")
        self.capture_vision = capture_vision

        self.n_act = int(model.action_dim) * int(model.num_action_chunks)
        self.vocab_size = int(getattr(model, "vocab_size", model.config.vocab_size))
        self.n_action_bins = int(model.config.n_action_bins)

        self._armed = False
        self._pending: Optional[CapturedCall] = None
        self._pending_logits: Optional[torch.Tensor] = None
        self._verify_next = False
        self._seq_len: Optional[int] = None
        self._n_hidden_layers: Optional[int] = None

        self._handles = [
            model.language_model.register_forward_hook(self._lm_hook, with_kwargs=True)
        ]
        self.vision_available = False
        if capture_vision:
            projector = getattr(model, "projector", None)
            if projector is None:
                # P5 is a stretch goal; a missing projector must not sink a run.
                self.vision_warning = (
                    "model has no `projector` attribute; vision capture disabled"
                )
            else:
                self.vision_available = True
                self._handles.append(
                    projector.register_forward_hook(self._projector_hook)
                )
        self._pending_vision: Optional[torch.Tensor] = None

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        """Remove every hook.  Safe to call twice."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def armed(self) -> "_ArmedContext":
        return _ArmedContext(self)

    def take(self) -> CapturedCall:
        """Pop the capture produced by the most recent armed forward."""
        if self._pending is None:
            raise RuntimeError(
                "no capture available: was the model called inside `armed()`?"
            )
        captured, self._pending = self._pending, None
        return captured

    # -- hooks -----------------------------------------------------------

    def _projector_hook(self, module, args, output) -> None:
        if not self._armed:
            return
        self._pending_vision = output.detach()

    def _lm_hook(self, module, args, kwargs, output) -> None:
        if not self._armed:
            return
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(
                "language model returned no hidden_states; expected "
                "predict_action_batch to pass output_hidden_states=True"
            )
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None and len(args) > 1:
            attention_mask = args[1]

        seq_len = hidden_states[-1].shape[1]
        self._seq_len = seq_len
        self._n_hidden_layers = len(hidden_states)
        split = seq_len - self.n_act - 1
        if split <= 1:
            raise RuntimeError(
                f"sequence of length {seq_len} is too short for {self.n_act} "
                "action tokens; check action_dim / num_action_chunks"
            )
        for layer in self.layers:
            if not 0 <= layer < len(hidden_states):
                raise RuntimeError(
                    f"capture.layers contains {layer}, but the model exposes "
                    f"{len(hidden_states)} hidden-state tensors (0..{len(hidden_states) - 1})"
                )

        pooled = self._pool(hidden_states, attention_mask, split)
        masked_action_logits = self.action_logits(output.logits, split)
        entropy_tokens = self._entropy_from_masked(masked_action_logits)
        if self._verify_next:
            self._pending_logits = masked_action_logits
            self._verify_next = False

        vision = None
        if self._pending_vision is not None:
            # [B, n_patches, d_proj] -> mean over patches.
            vision = (
                self._pending_vision.float()
                .mean(dim=1)
                .to("cpu", torch.float16)
                .numpy()
            )
            self._pending_vision = None

        self._pending = CapturedCall(
            hidden=pooled,
            entropy_mean=entropy_tokens.mean(axis=1).astype(np.float32),
            entropy_tokens=entropy_tokens,
            vision=vision,
            meta={
                "seq_len": int(seq_len),
                "n_hidden_layers": int(len(hidden_states)),
                "readout_start": int(split),
                "n_act": int(self.n_act),
            },
        )

    # -- pooling ---------------------------------------------------------

    def _pool(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        attention_mask: Optional[torch.Tensor],
        split: int,
    ) -> np.ndarray:
        """Reduce every requested layer to ``[B, n_pools, D]`` on CPU."""
        if attention_mask is not None:
            # float32, not the model dtype: the context is ~280 tokens long and
            # bfloat16 has a spacing of 2 in [256, 512), so a bf16 accumulator
            # would miscount the denominator by up to 1 -- a silent, per-task
            # scaling error in ctx_mean.
            ctx_mask = attention_mask[:, :split].to(torch.float32)
            # A fully-masked context would divide by zero; it never happens for
            # a real prompt, but guard rather than emit silent NaNs.
            ctx_denominator = ctx_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        else:
            ctx_mask = None
            ctx_denominator = None

        per_layer = []
        for layer in self.layers:
            states = hidden_states[layer]
            pooled = []
            for pool in self.pools:
                if pool == "act_mean":
                    value = states[:, split : split + self.n_act].float().mean(dim=1)
                elif pool == "ctx_last":
                    value = states[:, split].float()
                elif pool == "ctx_mean":
                    context = states[:, :split]
                    if ctx_mask is None:
                        value = context.float().mean(dim=1)
                    else:
                        weighted = context.float() * ctx_mask.unsqueeze(-1)
                        value = weighted.sum(dim=1) / ctx_denominator
                else:  # pragma: no cover - guarded in __init__
                    raise ValueError(f"unknown pool {pool!r}")
                pooled.append(value)
            per_layer.append(torch.stack(pooled, dim=1))  # [B, n_pools, D]

        stacked = torch.stack(per_layer, dim=1)  # [B, n_layers, n_pools, D]
        return stacked.detach().to("cpu", torch.float16).numpy()

    # -- action distribution ---------------------------------------------

    @property
    def bin_offset(self) -> int:
        """First vocabulary index that encodes an action bin."""
        return self.vocab_size - self.n_action_bins

    def action_logits(self, logits: torch.Tensor, split: int) -> torch.Tensor:
        """Slice out the logits that decode the action tokens.

        ``predict_action_batch`` masks everything outside the action-bin range
        to ``-inf`` and then softmaxes over the full 32k vocabulary.  Only
        ``n_action_bins`` (256) columns are ever finite, and a softmax over a
        vector padded with ``-inf`` is identical to the softmax over just the
        finite part -- so we keep the 256-wide core instead.  That is exact,
        not an approximation, and it turns a 143 MB fp32 temporary per call
        (times several more inside the entropy computation) into about 1 MB.

        Returned indices are relative to :attr:`bin_offset`.
        """
        # `copy=True` matters: predict_action_batch later masks
        # `outputs.logits` in place through a view, and we must not alias it.
        return logits[
            :, split : split + self.n_act, self.bin_offset : self.vocab_size
        ].to(dtype=torch.float32, copy=True)

    def _entropy_from_masked(self, masked: torch.Tensor) -> np.ndarray:
        """Per-token entropy of the *unscaled* action distribution.

        Deliberately computed at temperature 1 and without top-k, so the number
        measures the policy's own uncertainty rather than the sampling knobs in
        the run config.  ``masked`` holds only the finite action-bin columns,
        so no ``-inf`` reaches the ``p log p`` product and no NaN guard is
        needed.
        """
        logp = torch.log_softmax(masked, dim=-1)
        entropy = -(logp.exp() * logp).sum(dim=-1)
        return entropy.detach().to("cpu", torch.float32).numpy()

    # -- verification ----------------------------------------------------

    def arm_verification(self) -> None:
        """Ask the next armed forward to stash its sliced action logits."""
        self._verify_next = True

    def verify_indexing(
        self,
        action_tokens: torch.Tensor,
        model_logprobs: torch.Tensor,
        do_sample: bool,
        temperature: float,
        top_k: int,
        atol: float = 0.05,
        rtol: float = 0.02,
    ) -> dict[str, Any]:
        """Reproduce the model's own action logprobs from our slice.

        This is the decisive check that our ``[-n_act-1 : -1]`` window is the
        window the model decodes from.  A wrong offset shifts the distribution
        onto a different token and the mismatch is order 1, not order 0.01.

        Call :meth:`arm_verification` before the forward, then pass the
        ``action_tokens`` and ``prev_logprobs`` that ``predict_action_batch``
        returned.

        The tolerance is relative as well as absolute, because the model's own
        logprobs are computed in the logits' dtype.  Under bfloat16 one ulp at
        a logprob of -10 is already 0.0625, so a fixed 0.05 would abort a
        perfectly correct run the first time sampling drew a low-probability
        token.  A misaligned window produces errors of order 1 to 10, so the
        check keeps all of its discriminating power.
        """
        if self._pending_logits is None:
            raise RuntimeError(
                "no stashed action logits: call arm_verification() before the "
                "forward you want to verify"
            )
        masked, self._pending_logits = self._pending_logits, None
        if do_sample:
            masked = masked / temperature
            if top_k is not None and top_k > 0:
                kept = min(int(top_k), masked.shape[-1])
                threshold = masked.topk(kept, dim=-1).values[..., -1, None]
                masked = masked.masked_fill(masked < threshold, -math.inf)
        logp = torch.log_softmax(masked, dim=-1)
        # `masked` covers only the action-bin columns, so token ids have to be
        # shifted into that window before gathering.
        target = (
            action_tokens.reshape(masked.shape[0], self.n_act).to(logp.device)
            - self.bin_offset
        )
        if int(target.min()) < 0 or int(target.max()) >= masked.shape[-1]:
            raise RuntimeError(
                "action tokens fall outside the action-bin vocabulary range "
                f"[{self.bin_offset}, {self.vocab_size}) -- capture.py's view "
                "of the model's vocabulary layout is wrong"
            )
        ours = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        theirs = model_logprobs.reshape(ours.shape).to(ours.device).float()
        max_abs_diff = float((ours - theirs).abs().max().item())
        tolerance = atol + rtol * float(theirs.abs().max().item())
        result = {
            "max_abs_logprob_diff": max_abs_diff,
            "tolerance": tolerance,
            "logits_dtype": str(masked.dtype),
            "seq_len": int(self._seq_len),
            "readout_start": int(self._seq_len - self.n_act - 1),
            "passed": max_abs_diff <= tolerance,
        }
        if not result["passed"]:
            raise RuntimeError(
                "hidden-state indexing check failed: recomputed action "
                f"logprobs differ from the model's by {max_abs_diff:.4f} "
                f"(tolerance {tolerance:.4f}). The pool offsets in capture.py "
                "no longer match predict_action_batch -- re-derive them before "
                "collecting data."
            )
        return result


class _ArmedContext:
    """Context manager that turns the hooks on for exactly one forward."""

    def __init__(self, capture: HiddenStateCapture) -> None:
        self._capture = capture

    def __enter__(self) -> HiddenStateCapture:
        self._capture._armed = True
        self._capture._pending = None
        self._capture._pending_vision = None
        return self._capture

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._capture._armed = False
