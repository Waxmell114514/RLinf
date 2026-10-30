"""X6-EC: an efference-copy token for the ``rlinf`` OpenVLA-OFT class.

The official OFT class appends one proprio token after the image patches; the
``rlinf`` class, which loads every checkpoint behind E4, X1, X4 and the X6
pilot, builds its embedding from vision and text only, and the two classes
agree on only 95.7% of greedy tokens (logbook, X6 stage 0 step 2).  So every
X6-EC arm runs through the ``rlinf`` class, and this module adds the token to
it from outside ``rlinf/``:

- :func:`adopt` turns a loaded ``rlinf``-class model into a subclass that owns
  an :class:`EfferenceProjector` (the official ``ProprioProjector`` shape:
  Linear, GELU, Linear; the last layer starts at zero);
- its ``_build_embedding`` calls the parent's and inserts the projected input
  right after the patches, where the official class puts its proprio token;
- the parent slices action logits by an offset counted from the start of the
  sequence in ``predict_action_batch`` and from the end in ``default_forward``.
  A forward hook on the language model drops the token's own logit row, so
  both slices line up unchanged; hidden states keep the row, and the parent
  reads them from the end only.

The input reaches the model as ``env_obs["efference"]`` (rollout) or
``forward_inputs["efference"]`` (training replay), shape [B, in_dim].
:class:`EfferenceHistory` builds it per env: the normalised state, then for
each of the last K chunks the sent command and the resulting state change,
zero-padded at episode start (136-d for K = 2, chunk 8; 8-d for state only).

Frozen arm F is the plain parent class; an adopted model refuses to run
without an input, so a missing history cannot pass silently as arm F.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch import nn

KINDS = ("copy", "state")
STATE_DIM = 8
ACTION_DIM = 7
DELTA_CLIP = 10.0  # scaled state changes; bounds an axis-angle wrap


def efference_dim(
    kind: str,
    history: int = 2,
    chunk: int = 8,
    action_dim: int = ACTION_DIM,
    state_dim: int = STATE_DIM,
) -> int:
    """Input width: the state, plus K x (sent chunk, state change) for a copy."""
    if kind not in KINDS:
        raise ValueError(f"efference kind must be one of {KINDS}, got {kind!r}")
    if kind == "state":
        return state_dim
    return state_dim + history * (chunk * action_dim + state_dim)


class EfferenceProjector(nn.Module):
    """``prismatic``'s ProprioProjector layout, with the output layer at zero.

    At zero the token is the zero vector for every input: the input cannot
    move the actions until training moves ``fc2``.  (A zero token is still a
    token: it takes attention mass and shifts later positions by one, so an
    untrained adopted model is not the parent.)
    """

    def __init__(self, in_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.llm_dim = llm_dim
        self.fc1 = nn.Linear(in_dim, llm_dim, bias=True)
        self.fc2 = nn.Linear(llm_dim, llm_dim, bias=True)
        self.act_fn1 = nn.GELU()
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn1(self.fc1(x)))


class EfferenceMixin:
    """Overrides for the ``rlinf`` OFT class; see the module docstring."""

    efference_projector: EfferenceProjector

    # -- input plumbing ---------------------------------------------------

    def _efference_tensor(self, value: Any) -> torch.Tensor:
        if value is None:
            raise ValueError(
                "an efference-adopted model needs an efference input on every "
                "call (env_obs['efference'] or forward_inputs['efference'])"
            )
        param = next(self.efference_projector.parameters())
        x = torch.as_tensor(value).to(device=param.device, dtype=param.dtype)
        if x.ndim != 2 or x.shape[1] != self.efference_projector.in_dim:
            raise ValueError(
                f"efference input must be [B, {self.efference_projector.in_dim}], "
                f"got {tuple(x.shape)}"
            )
        return x

    @contextmanager
    def _efference_input(self, x: torch.Tensor):
        self._efference_x = x
        try:
            yield
        finally:
            self._efference_x = None
            self._efference_row = None

    def _num_patch_tokens(self) -> int:
        return (
            self.vision_backbone.get_num_patches()
            * self.vision_backbone.get_num_images_in_input()
        )

    # -- the token ----------------------------------------------------------

    def _build_embedding(self, input_ids, attention_mask, pixel_values):
        embeddings, mask = super()._build_embedding(
            input_ids, attention_mask, pixel_values
        )
        x = getattr(self, "_efference_x", None)
        if x is None:
            return embeddings, mask
        if x.shape[0] != embeddings.shape[0]:
            raise ValueError(
                f"efference batch {x.shape[0]} != model batch {embeddings.shape[0]}"
            )
        # [BOS, patches, text...]: the parent inserts the patches after BOS.
        row = 1 + self._num_patch_tokens()
        token = self.efference_projector(x.to(embeddings.dtype)).unsqueeze(1)
        embeddings = torch.cat([embeddings[:, :row], token, embeddings[:, row:]], 1)
        mask = torch.cat(
            [mask[:, :row], torch.ones_like(mask[:, :1]), mask[:, row:]], 1
        )
        self._efference_row = row
        return embeddings, mask

    def _drop_efference_logit(self, module, args, output):
        """Forward hook: remove the token's logit row (see module docstring)."""
        row = getattr(self, "_efference_row", None)
        if row is None or getattr(output, "logits", None) is None:
            return output
        self._efference_row = None
        logits = output.logits
        output.logits = torch.cat([logits[:, :row], logits[:, row + 1 :]], dim=1)
        return output

    # -- entry points -------------------------------------------------------

    @torch.no_grad()
    def predict_action_batch(self, *args, env_obs=None, efference=None, **kwargs):
        if efference is None and env_obs is not None:
            efference = env_obs.get("efference")
        x = self._efference_tensor(efference)
        with self._efference_input(x):
            actions, result = super().predict_action_batch(
                *args, env_obs=env_obs, **kwargs
            )
        # Training replays the call through default_forward with the same input.
        result["forward_inputs"]["efference"] = x
        return actions, result

    def default_forward(self, *args, forward_inputs=None, efference=None, **kwargs):
        if efference is None and forward_inputs is not None:
            efference = forward_inputs.get("efference")
        x = self._efference_tensor(efference)
        with self._efference_input(x):
            return super().default_forward(
                *args, forward_inputs=forward_inputs, **kwargs
            )

    # -- weights --------------------------------------------------------------

    def efference_state(self) -> dict[str, Any]:
        return {
            "in_dim": self.efference_projector.in_dim,
            "projector": {
                k: v.detach().cpu()
                for k, v in self.efference_projector.state_dict().items()
            },
        }

    def load_efference_state(self, state: dict[str, Any]) -> None:
        if state["in_dim"] != self.efference_projector.in_dim:
            raise ValueError(
                f"saved projector takes {state['in_dim']}-d input, "
                f"model has {self.efference_projector.in_dim}"
            )
        self.efference_projector.load_state_dict(state["projector"])


_SUBCLASSES: dict[type, type] = {}


def efference_class(base: type) -> type:
    """``EfferenceMixin`` over ``base``, created once per base class."""
    if base not in _SUBCLASSES:
        _SUBCLASSES[base] = type(
            f"Efference{base.__name__}", (EfferenceMixin, base), {}
        )
    return _SUBCLASSES[base]


def adopt(model: nn.Module, in_dim: int, seed: int = 0) -> nn.Module:
    """Give a loaded ``rlinf``-class OFT model the efference token, in place.

    The projector's first layer is drawn from ``seed`` (same seed, same draw
    for arms of the same width); its last layer starts at zero.
    """
    if isinstance(model, EfferenceMixin):
        raise ValueError("model already has an efference token")
    param = next(model.parameters())
    llm_dim = model.get_input_embeddings().weight.shape[1]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        projector = EfferenceProjector(in_dim, llm_dim)
    model.__class__ = efference_class(type(model))
    model.efference_projector = projector.to(device=param.device, dtype=param.dtype)
    model._efference_x = None
    model._efference_row = None
    model._efference_hook = model.language_model.register_forward_hook(
        model._drop_efference_logit
    )
    return model


# -- the input ------------------------------------------------------------------


def normalize_state(
    states: np.ndarray, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    """``prismatic``'s BOUNDS_Q99 proprio normalisation: [low, high] -> [-1, 1]."""
    scaled = 2.0 * (states - low) / (high - low + 1e-8) - 1.0
    return np.clip(scaled, -1.0, 1.0)


class EfferenceHistory:
    """Per-env input builder: state, then K x (sent chunk, state change).

    ``vector(states)`` gives the input for the current call; ``push`` records
    what was sent this call (before any actuator fault: the copy is of the
    command, not of its effect) and the state change it produced.  Slots are
    zero-padded until K chunks have run; ``reset(slots)`` starts new episodes.
    """

    def __init__(
        self,
        num_envs: int,
        kind: str,
        state_low: Sequence[float],
        state_high: Sequence[float],
        delta_scale: Optional[Sequence[float]] = None,
        history: int = 2,
        chunk: int = 8,
        action_dim: int = ACTION_DIM,
    ) -> None:
        self.kind = kind
        self.dim = efference_dim(kind, history, chunk, action_dim, len(state_low))
        self.state_low = np.asarray(state_low, dtype=np.float32)
        self.state_high = np.asarray(state_high, dtype=np.float32)
        if self.state_low.shape != self.state_high.shape:
            raise ValueError("state_low and state_high differ in length")
        state_dim = self.state_low.shape[0]
        if kind == "copy":
            if delta_scale is None or len(delta_scale) != state_dim:
                raise ValueError(f"a copy needs delta_scale of length {state_dim}")
            self.delta_scale = np.asarray(delta_scale, dtype=np.float32)
            if (self.delta_scale <= 0).any():
                raise ValueError("delta_scale must be positive")
        self.history = history
        self.chunk = chunk
        self.action_dim = action_dim
        # Newest first: [:, 0] is chunk c-1, [:, 1] is c-2.
        self._sent = np.zeros((num_envs, history, chunk * action_dim), np.float32)
        self._delta = np.zeros((num_envs, history, state_dim), np.float32)

    def reset(self, slots: Optional[Sequence[int]] = None) -> None:
        index = slice(None) if slots is None else np.asarray(slots)
        self._sent[index] = 0.0
        self._delta[index] = 0.0

    def vector(self, states: np.ndarray) -> np.ndarray:
        state = normalize_state(
            np.asarray(states, dtype=np.float32), self.state_low, self.state_high
        )
        if self.kind == "state":
            return state.astype(np.float32)
        pieces = [state]
        for k in range(self.history):
            pieces += [self._sent[:, k], self._delta[:, k]]
        return np.concatenate(pieces, axis=1).astype(np.float32)

    def push(
        self,
        sent: np.ndarray,
        states_before: np.ndarray,
        states_after: np.ndarray,
    ) -> None:
        if self.kind == "state":
            return
        sent = np.asarray(sent, dtype=np.float32).reshape(len(self._sent), -1)
        if sent.shape[1] != self.chunk * self.action_dim:
            raise ValueError(
                f"sent chunk has {sent.shape[1]} values, expected "
                f"{self.chunk * self.action_dim}"
            )
        delta = (
            np.asarray(states_after) - np.asarray(states_before)
        ) / self.delta_scale
        self._sent = np.roll(self._sent, 1, axis=1)
        self._delta = np.roll(self._delta, 1, axis=1)
        self._sent[:, 0] = sent
        self._delta[:, 0] = np.clip(delta, -DELTA_CLIP, DELTA_CLIP)
