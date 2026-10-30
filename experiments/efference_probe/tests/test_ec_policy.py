"""X6-EC: the efference token on the rlinf OFT class (ec_policy.py).

``prismatic`` is not installed here, so the parent is a small fake that keeps
the rlinf class's sequence layout and copies its three index expressions
verbatim (``rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py``):
the patch insertion after BOS, the action-logit slice of
``predict_action_batch`` (counted from the start) and of ``default_forward``
(counted from the end).  ``x6_ec_smoke.py`` repeats the agreement check on
the real class and checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efference_probe import ec_policy  # noqa: E402

STOP = 2
D = 16
VOCAB = 40
N_PATCHES = 4
ACTION_DIM = 2
CHUNKS = 3
ACT = ACTION_DIM * CHUNKS
PROMPT = 5  # BOS + 3 tokens + the trailing space


class FakeBackbone:
    def get_num_patches(self):
        return N_PATCHES

    def get_num_images_in_input(self):
        return 1


class FakeLM(nn.Module):
    """Causal and position-aware, so a token or a shift changes later rows."""

    def __init__(self):
        super().__init__()
        self.pos = nn.Embedding(64, D)
        # Rows differ by O(1) from their neighbours, as in a real LM.
        self.pos_out = nn.Embedding(64, D)
        self.mix = nn.Linear(D, D)
        self.head = nn.Linear(D, VOCAB)

    def forward(self, inputs_embeds, attention_mask, position_ids, **kwargs):
        x = (inputs_embeds + self.pos(position_ids)) * attention_mask[..., None]
        count = attention_mask.cumsum(1).clamp(min=1)[..., None]
        hidden = torch.tanh(
            self.mix(x.cumsum(1) / count) + 3 * self.pos_out(position_ids)
        )
        return SimpleNamespace(logits=self.head(hidden), hidden_states=(hidden,))


class FakeOFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_dim, self.num_action_chunks = ACTION_DIM, CHUNKS
        self.vision_backbone = FakeBackbone()
        self.embed = nn.Embedding(VOCAB, D)
        self.vision = nn.Linear(3, N_PATCHES * D)
        self.language_model = FakeLM()

    def get_input_embeddings(self):
        return self.embed

    def _prepare_input_for_action_prediction(self, input_ids, attention_mask):
        b = input_ids.shape[0]
        ids = torch.cat(
            [
                input_ids,
                torch.ones(b, ACT, dtype=input_ids.dtype),
                torch.full((b, 1), STOP, dtype=input_ids.dtype),
            ],
            1,
        )
        mask = torch.cat(
            [attention_mask, torch.ones(b, ACT + 1, dtype=attention_mask.dtype)], 1
        )
        return ids, mask

    def _build_embedding(self, input_ids, attention_mask, pixel_values):
        input_ids, attention_mask = input_ids[:, :-1], attention_mask[:, :-1]
        actions_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        actions_mask[:, -ACT:] = True
        embeddings = self.embed(input_ids) * (~actions_mask[..., None])
        patches = self.vision(pixel_values).reshape(len(input_ids), N_PATCHES, D)
        # prismatic's _build_multimodal_attention: patches go after <BOS>.
        mm = torch.cat([embeddings[:, :1], patches, embeddings[:, 1:]], 1)
        mask = torch.cat(
            [
                attention_mask[:, :1],
                torch.ones(len(input_ids), N_PATCHES, dtype=attention_mask.dtype),
                attention_mask[:, 1:],
            ],
            1,
        )
        return mm, mask

    def _run_lm(self, input_ids, attention_mask, pixel_values):
        input_ids, attention_mask = self._prepare_input_for_action_prediction(
            input_ids, attention_mask
        )
        mm_embeddings, mm_attention_mask = self._build_embedding(
            input_ids, attention_mask, pixel_values
        )
        outputs = self.language_model(
            inputs_embeds=mm_embeddings,
            attention_mask=mm_attention_mask,
            position_ids=mm_attention_mask.cumsum(dim=1) - 1,
        )
        assert outputs.hidden_states[-1].shape[1] == mm_embeddings.shape[1]
        return outputs

    @torch.no_grad()
    def predict_action_batch(
        self, input_ids=None, attention_mask=None, pixel_values=None, env_obs=None, **kw
    ):
        if env_obs is not None:
            input_ids = env_obs["input_ids"]
            attention_mask = env_obs["attention_mask"]
            pixel_values = env_obs["pixel_values"]
        n_prompt_tokens = input_ids.shape[-1] - 1
        n_patches = (
            self.vision_backbone.get_num_patches()
            * self.vision_backbone.get_num_images_in_input()
        )
        outputs = self._run_lm(input_ids, attention_mask, pixel_values)
        logits_tensor = outputs.logits[
            :,
            n_patches + n_prompt_tokens : n_patches
            + n_prompt_tokens
            + self.action_dim * self.num_action_chunks,
            :,
        ]
        idxs = logits_tensor.argmax(dim=-1)
        logprobs = torch.log_softmax(logits_tensor, -1).gather(-1, idxs[..., None])
        forward_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "action_tokens": idxs,
        }
        return idxs.float(), {
            "prev_logprobs": logprobs[..., 0],
            "forward_inputs": forward_inputs,
        }

    def default_forward(self, forward_inputs=None, **kw):
        outputs = self._run_lm(
            forward_inputs["input_ids"],
            forward_inputs["attention_mask"],
            forward_inputs["pixel_values"],
        )
        logits = outputs.logits[:, -self.action_dim * self.num_action_chunks - 1 : -1]
        tokens = forward_inputs["action_tokens"]
        return {
            "logprobs": torch.log_softmax(logits, -1).gather(-1, tokens[..., None])[
                ..., 0
            ]
        }


def _batch(b=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, VOCAB, (b, PROMPT), generator=g)
    ids[:, 0] = 1
    return {
        "input_ids": ids,
        "attention_mask": torch.ones(b, PROMPT, dtype=torch.long),
        "pixel_values": torch.randn(b, 3, generator=g),
    }


def _model(seed=0):
    torch.manual_seed(seed)
    return FakeOFT().eval()


def _randomise_fc2(model, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.efference_projector.fc2.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.5)


def test_predict_and_forward_score_the_same_tokens_with_the_token():
    model = ec_policy.adopt(_model(), in_dim=136)
    _randomise_fc2(model)
    obs = dict(_batch(), efference=np.random.default_rng(0).normal(size=(3, 136)))
    _, result = model.predict_action_batch(env_obs=obs)
    replay = model.default_forward(forward_inputs=result["forward_inputs"])
    torch.testing.assert_close(replay["logprobs"], result["prev_logprobs"])
    assert result["forward_inputs"]["efference"].shape == (3, 136)


def test_without_the_hook_the_two_slices_disagree():
    """The check above has teeth: the start-counted slice is off by one."""
    model = ec_policy.adopt(_model(), in_dim=136)
    _randomise_fc2(model)
    model._efference_hook.remove()
    obs = dict(_batch(), efference=np.random.default_rng(0).normal(size=(3, 136)))
    _, result = model.predict_action_batch(env_obs=obs)
    replay = model.default_forward(forward_inputs=result["forward_inputs"])
    assert not torch.allclose(replay["logprobs"], result["prev_logprobs"])


def test_token_sits_after_the_patches_and_reads_the_input():
    model = ec_policy.adopt(_model(), in_dim=8)
    _randomise_fc2(model)
    batch = _batch()
    x = torch.randn(3, 8)
    ids, mask = model._prepare_input_for_action_prediction(
        batch["input_ids"], batch["attention_mask"]
    )
    plain, _ = FakeOFT._build_embedding(model, ids, mask, batch["pixel_values"])
    with model._efference_input(x):
        mm, mm_mask = model._build_embedding(ids, mask, batch["pixel_values"])
    row = 1 + N_PATCHES
    assert mm.shape[1] == plain.shape[1] + 1 and mm_mask.shape == mm.shape[:2]
    torch.testing.assert_close(mm[:, :row], plain[:, :row])
    torch.testing.assert_close(mm[:, row], model.efference_projector(x))
    torch.testing.assert_close(mm[:, row + 1 :], plain[:, row:])
    assert (mm_mask[:, row] == 1).all()


def test_zero_init_ignores_the_input_and_trained_does_not():
    model = ec_policy.adopt(_model(), in_dim=136)
    batch = _batch()
    rng = np.random.default_rng(0)
    first, second = rng.normal(size=(3, 136)), rng.normal(size=(3, 136))
    _, a = model.predict_action_batch(env_obs=dict(batch, efference=first))
    _, b = model.predict_action_batch(env_obs=dict(batch, efference=second))
    torch.testing.assert_close(a["prev_logprobs"], b["prev_logprobs"])
    _randomise_fc2(model)
    with torch.no_grad():
        model.efference_projector.fc1.weight.mul_(5)
    outs = [
        model.default_forward(
            forward_inputs=dict(
                a["forward_inputs"], efference=torch.as_tensor(x).float()
            )
        )["logprobs"]
        for x in (first, second)
    ]
    assert not torch.allclose(*outs)


def test_inputs_are_required_and_checked():
    model = ec_policy.adopt(_model(), in_dim=136)
    with pytest.raises(ValueError, match="needs an efference input"):
        model.predict_action_batch(env_obs=_batch())
    with pytest.raises(ValueError, match=r"\[B, 136\]"):
        model.predict_action_batch(env_obs=dict(_batch(), efference=np.zeros((3, 8))))
    with pytest.raises(ValueError, match="batch"):
        model.predict_action_batch(env_obs=dict(_batch(), efference=np.zeros((2, 136))))
    with pytest.raises(ValueError, match="already"):
        ec_policy.adopt(model, in_dim=136)
    # State is cleared even after a failed call.
    assert model._efference_x is None and model._efference_row is None


def test_adoption_is_per_instance_and_seeded():
    plain, adopted = _model(), ec_policy.adopt(_model(), in_dim=136, seed=3)
    assert type(plain) is FakeOFT and isinstance(adopted, FakeOFT)
    _, result = plain.predict_action_batch(env_obs=_batch())
    assert "efference" not in result["forward_inputs"]
    again = ec_policy.adopt(_model(), in_dim=136, seed=3)
    torch.testing.assert_close(
        adopted.efference_projector.fc1.weight, again.efference_projector.fc1.weight
    )
    assert type(adopted) is type(again)  # one subclass per base
    assert adopted.efference_projector.fc2.weight.abs().sum() == 0


def test_projector_state_round_trips_and_checks_width():
    source = ec_policy.adopt(_model(), in_dim=136)
    _randomise_fc2(source)
    state = source.efference_state()
    target = ec_policy.adopt(_model(), in_dim=136, seed=9)
    target.load_efference_state(state)
    for key, value in state["projector"].items():
        torch.testing.assert_close(target.efference_projector.state_dict()[key], value)
    with pytest.raises(ValueError, match="136-d"):
        ec_policy.adopt(_model(), in_dim=8).load_efference_state(state)


# -- the input builder ----------------------------------------------------------

LOW, HIGH = -np.ones(8), np.ones(8)
SCALE = np.full(8, 0.5)


def test_dims_match_the_prereg():
    assert ec_policy.efference_dim("copy") == 136
    assert ec_policy.efference_dim("state") == 8
    assert ec_policy.efference_dim("copy", history=1, chunk=5) == 8 + 43
    with pytest.raises(ValueError):
        ec_policy.efference_dim("shuffled")


def test_history_is_zero_padded_then_newest_first():
    hist = ec_policy.EfferenceHistory(2, "copy", LOW, HIGH, SCALE)
    states = np.zeros((2, 8))
    v0 = hist.vector(states)
    assert v0.shape == (2, 136) and not v0[:, 8:].any()
    sent1 = np.full((2, 8, 7), 0.1)
    hist.push(sent1, states, states + 0.5)
    v1 = hist.vector(states)
    np.testing.assert_allclose(v1[:, 8:64], 0.1, rtol=1e-6)
    np.testing.assert_allclose(v1[:, 64:72], 1.0)  # 0.5 / scale 0.5
    assert not v1[:, 72:].any()
    sent2 = np.full((2, 8, 7), -0.2)
    hist.push(sent2, states, states - 0.25)
    v2 = hist.vector(states)
    np.testing.assert_allclose(v2[:, 8:64], -0.2, rtol=1e-6)
    np.testing.assert_allclose(v2[:, 64:72], -0.5)
    np.testing.assert_allclose(v2[:, 72:128], 0.1, rtol=1e-6)
    np.testing.assert_allclose(v2[:, 128:136], 1.0)
    hist.reset([1])
    v3 = hist.vector(states)
    assert v3[0, 8:].any() and not v3[1, 8:].any()


def test_state_is_normalised_and_deltas_clipped():
    hist = ec_policy.EfferenceHistory(1, "copy", np.zeros(8), np.full(8, 2.0), SCALE)
    v = hist.vector(np.array([[0.0, 1.0, 2.0, 3.0, -1.0, 0.5, 1.5, 1.0]]))
    np.testing.assert_allclose(v[0, :8], [-1, 0, 1, 1, -1, -0.5, 0.5, 0], atol=1e-6)
    hist.push(np.zeros((1, 56)), np.zeros((1, 8)), np.full((1, 8), 100.0))
    assert hist.vector(np.zeros((1, 8)))[0, 64:72].max() == ec_policy.DELTA_CLIP


def test_state_only_ignores_pushes_and_copy_needs_scales():
    hist = ec_policy.EfferenceHistory(2, "state", LOW, HIGH)
    hist.push(np.ones((2, 56)), np.zeros((2, 8)), np.ones((2, 8)))
    assert hist.vector(np.zeros((2, 8))).shape == (2, 8)
    with pytest.raises(ValueError, match="delta_scale"):
        ec_policy.EfferenceHistory(2, "copy", LOW, HIGH)
    with pytest.raises(ValueError, match="positive"):
        ec_policy.EfferenceHistory(2, "copy", LOW, HIGH, np.zeros(8))
    hist = ec_policy.EfferenceHistory(2, "copy", LOW, HIGH, SCALE)
    with pytest.raises(ValueError, match="expected 56"):
        hist.push(np.ones((2, 40)), np.zeros((2, 8)), np.ones((2, 8)))


def test_smoke_agreement_check_runs_on_the_fake():
    from efference_probe import x6_ec_smoke

    model = _model()
    obs = _batch()
    plain = x6_ec_smoke._agreement(model, obs)
    ec_policy.adopt(model, in_dim=136)
    _randomise_fc2(model)
    adopted = x6_ec_smoke._agreement(model, obs, np.zeros((3, 136), np.float32))
    assert plain["agrees"] and adopted["agrees"] and adopted["actions_finite"]
    model._efference_hook.remove()
    assert not x6_ec_smoke._agreement(model, obs, np.ones((3, 136), np.float32))[
        "agrees"
    ]
