"""X6-EC end to end on CPU fakes: collection, samples, training, evaluation, rules.

The collection runs the real harness loop on the fake env of
``test_x1_x4_harness``; training runs ``train_ec.train`` with a fake OFT that
keeps the real class's sequence layout (``test_ec_policy``) and real LoRA
(peft).  The trained arm is then loaded the way the harness loads it and
evaluated, and the inputs it received online are checked against the offline
replay the trainer's samples come from.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("omegaconf")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efference_probe import ec_dataset, ec_policy, faults, train_ec  # noqa: E402
from efference_probe import x6_ec_report as report  # noqa: E402
from efference_probe.config import load_config  # noqa: E402
from efference_probe.tests import test_ec_policy as fake_oft  # noqa: E402
from efference_probe.tests import test_x1_x4_harness as fake  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CONFIG = CONFIGS / "main02_mirror.yaml"
N_ENVS, CHUNKS, DIM = fake.N_ENVS, fake.CHUNKS, fake.DIM
VOCAB = fake_oft.VOCAB
_EDGES = np.linspace(-1, 1, 16)
CENTERS = (_EDGES[:-1] + _EDGES[1:]) / 2
MASK = [True] * 6 + [False]
TOKENIZER = ec_dataset.ActionTokenizer(VOCAB, CENTERS, -np.ones(7), np.ones(7), MASK)
LOWEST_TOKEN = VOCAB - len(CENTERS)


# -- faults -------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec", ["bias:0.3@45", "bias:0.25@200", "bias_x:0.3", "invert_x", "invert_y"]
)
def test_the_oracle_inverts_each_fault_inside_the_bounds(spec):
    fault = faults.parse(spec)
    command = np.random.default_rng(0).uniform(-0.5, 0.5, size=(3, CHUNKS, DIM))
    command = command.astype(np.float32)
    np.testing.assert_allclose(
        fault.apply(fault.compensate(command)), command, atol=1e-6
    )
    changed = np.flatnonzero(
        (np.abs(fault.apply(command) - command) > 1e-9).any(axis=(0, 1))
    )
    assert tuple(changed) == fault.dims


def test_2d_bias_matches_bias_x_on_the_axis_and_bad_specs_are_refused():
    chunk = np.random.default_rng(1).uniform(-0.6, 0.6, size=(CHUNKS, DIM))
    np.testing.assert_allclose(
        faults.parse("bias:0.3@0").apply(chunk),
        faults.parse("bias_x:0.3").apply(chunk),
        atol=1e-6,
    )
    y = faults.parse("bias:0.2@90").apply(chunk) - chunk
    np.testing.assert_allclose(y[:, :2], np.tile([0.0, 0.2], (CHUNKS, 1)), atol=1e-6)
    for bad in ("bias:0.3", "bias:1.2@0", "invert_y:1", "bias:x@0"):
        with pytest.raises(ValueError):
            faults.parse(bad)


def test_training_draws_follow_the_prereg():
    rng = np.random.default_rng(0)
    draws = [faults.draw_training_fault(rng) for _ in range(20000)]
    family = np.array([d.fault.family for d in draws])
    comp = np.array([d.compensated for d in draws])
    assert abs(np.mean(family == "clean") - 0.30) < 0.015
    assert abs(np.mean(family == "bias") - 0.45) < 0.015
    assert abs(np.mean(np.isin(family, ["invert_x", "invert_y"])) - 0.25) < 0.015
    assert not comp[family == "clean"].any()
    assert abs(comp[family != "clean"].mean() - 0.70) < 0.015
    biases = [d.fault for d in draws if d.fault.family == "bias"]
    angles = np.array([f.angle_deg for f in biases])
    to_axis = np.abs(((angles + 45) % 90) - 45)
    assert to_axis.min() >= 10.0 and to_axis.max() <= 45.0
    norms = np.array([f.magnitude for f in biases])
    assert norms.min() >= 0.10 and norms.max() <= 0.35
    again = np.random.default_rng(0)
    assert [faults.draw_training_fault(again) for _ in range(50)] == draws[:50]


# -- config ---------------------------------------------------------------------


def test_new_sections_leave_earlier_hashes_and_refuse_bad_combinations():
    # The hashes every earlier run logged (X6 configs as committed in eff0d94).
    assert load_config(str(CONFIG)).hash() == "17cf313154d1"
    assert load_config(str(CONFIGS / "x6_fault_oft.yaml")).hash() == "ae1a69dee999"
    collect = load_config(str(CONFIGS / "x6_ec_collect.yaml"))
    assert collect.fault.mode == "draw" and collect.env.init_state_offset == 30
    assert collect.env.init_state_offset >= 30  # never an evaluation state
    evaluate = load_config(str(CONFIGS / "x6_ec_eval.yaml"))
    assert evaluate.fault.sweep == ["clean", "bias_x:0.3", "invert_x"]
    off = [
        "hijack.enabled=false",
        "capture.enabled=false",
        "capture.verify_indexing=false",
    ]
    bad = {
        "fault.mode 'draw' takes no sweep": [
            "fault.enabled=true",
            "fault.mode=draw",
            "fault.sweep=[clean]",
        ],
        "passes applies": [
            "fault.enabled=true",
            "fault.sweep=[clean]",
            "fault.passes=2",
        ],
        "unknown fault.mode": ["fault.enabled=true", "fault.mode=mixed"],
        "needs efference.arm_dir": ["efference.enabled=true"],
        "training collection runs the frozen": [
            "efference.enabled=true",
            "efference.arm_dir=x",
            "fault.enabled=true",
            "fault.mode=draw",
        ],
        "init_state_offset": ["env.init_state_offset=-1"],
    }
    for message, overrides in bad.items():
        with pytest.raises(ValueError, match=message):
            load_config(str(CONFIG), off + overrides)
    with pytest.raises(ValueError, match="capture, hijack and monitor"):
        load_config(str(CONFIG), ["efference.enabled=true", "efference.arm_dir=x"])


# -- tokens and samples -----------------------------------------------------------


def test_tokenizer_inverts_the_model_decoding():
    tokens = np.arange(LOWEST_TOKEN, VOCAB)
    values = TOKENIZER.values(np.tile(tokens[:, None], (1, 7)))
    np.testing.assert_array_equal(
        TOKENIZER.tokens(values)[:, :6], values[:, :6] * 0 + tokens[:, None]
    )
    real = ec_dataset.ActionTokenizer(
        32000, (np.linspace(-1, 1, 256)[:-1] + np.linspace(-1, 1, 256)[1:]) / 2,
        [-0.9375] * 6 + [0.0], [0.9375] * 6 + [1.0], MASK,
    )  # fmt: skip
    ids = np.random.default_rng(0).integers(32000 - 255, 32000, size=(50, 8, 7))
    np.testing.assert_array_equal(real.tokens(real.values(ids))[..., :6], ids[..., :6])
    v = np.random.default_rng(1).uniform(-0.9, 0.9, size=(100, 7))
    half_bin = 0.9375 * (CENTERS[1] - CENTERS[0]) / 2 + 1e-9
    assert np.abs(real.values(real.tokens(v)) - v)[:, :6].max() <= 0.9375 / 255 + 1e-6
    assert half_bin > 0


def test_targets_replace_only_the_faulted_dims():
    own = np.random.default_rng(0).integers(LOWEST_TOKEN, VOCAB, size=(CHUNKS, DIM))
    command = TOKENIZER.values(own)
    np.testing.assert_array_equal(
        ec_dataset.target_tokens(own, command, "clean", TOKENIZER), own
    )
    fault = faults.parse("invert_x")
    target = ec_dataset.target_tokens(
        own, fault.compensate(command), "invert_x", TOKENIZER
    )
    np.testing.assert_array_equal(target[:, 1:], own[:, 1:])
    np.testing.assert_array_equal(target[:, 0], TOKENIZER.tokens(-command)[:, 0])
    bias = ec_dataset.target_tokens(own, command, "bias:0.2@30", TOKENIZER)
    np.testing.assert_array_equal(bias[:, 2:], own[:, 2:])


def _calls(rows):
    out = pd.DataFrame(rows)
    out["episode"] = out["episode_id"]
    return out


def test_stats_scale_by_clean_changes_and_widen_flat_dims():
    rng = np.random.default_rng(0)
    rows = []
    for episode, fault in enumerate(["clean", "clean", "invert_x"]):
        for call in range(5):
            before = rng.normal(size=8)
            before[7] = 0.5  # a flat dim
            step = rng.normal(scale=0.1 if fault == "clean" else 5.0, size=8)
            rows.append(
                {
                    "episode_id": episode,
                    "call_idx": call,
                    "fault": fault,
                    "states_before": before,
                    "states_after": before + step,
                }
            )
    stats = ec_dataset.fit_input_stats(_calls(rows))
    assert max(stats["delta_scale"]) < 0.5  # the faulted episode is left out
    assert stats["state_high"][7] - stats["state_low"][7] >= 1e-3 - 1e-6
    assert stats["state_bounds_source"] == "collection"
    bounds = ([-1.0] * 8, [1.0] * 8)
    assert ec_dataset.fit_input_stats(_calls(rows), bounds)["state_low"] == [-1.0] * 8
    with pytest.raises(ValueError, match="no clean"):
        ec_dataset.fit_input_stats(_calls([r for r in rows if r["fault"] != "clean"]))


def test_shuffled_history_comes_from_another_episode():
    n = 60
    samples = ec_dataset.Samples(
        efference=np.arange(n * 136, dtype=np.float32).reshape(n, 136),
        targets=np.zeros((n, CHUNKS, DIM), np.int64),
        delta_target=np.zeros((n, 8), np.float32),
        a_sent=np.zeros((n, 56), np.float32),
        states=np.zeros((n, 8), np.float32),
        episode=np.repeat(np.arange(6), 10),
        frames=[""] * n,
        descriptions=[""] * n,
    )
    shuffled = ec_dataset.shuffled_history(samples, np.random.default_rng(0))
    np.testing.assert_array_equal(shuffled[:, :8], samples.efference[:, :8])
    donor = (shuffled[:, 8] - 8) // 136
    assert (samples.episode[donor.astype(int)] != samples.episode).all()


# -- the collection loop ---------------------------------------------------------


class TokenModel:
    """The frozen policy of the collection: tokens first, actions decoded."""

    def __init__(self):
        self.calls = 0

    def predict_action_batch(self, env_obs, **_sampling):
        rng = np.random.default_rng(self.calls)
        self.calls += 1
        tokens = rng.integers(LOWEST_TOKEN, VOCAB, size=(N_ENVS, CHUNKS, DIM))
        actions = TOKENIZER.values(tokens).astype(np.float32)
        actions[..., 6] = np.where(rng.random((N_ENVS, CHUNKS)) < 0.5, -1.0, 1.0)
        result = {
            "prev_logprobs": torch.zeros(N_ENVS, CHUNKS * DIM),
            "forward_inputs": {"action_tokens": torch.from_numpy(tokens)},
        }
        return torch.from_numpy(actions), result


class PromptEnv(fake.FakeEnv):
    cumsum_trial_id_bins = np.array([50, 100, 150, 200])

    def _obs(self):
        obs = super()._obs()
        obs["task_descriptions"] = ["put the bowl on the plate"] * N_ENVS
        return obs


def _collect_run(root: Path, passes: int = 3) -> Path:
    harness = fake._harness(
        root,
        [
            "run_id=collect",
            "hijack.enabled=false",
            "capture.enabled=false",
            "frames.mode=all",
            "frames.save_wrist=false",
            "fault.enabled=true",
            "fault.mode=draw",
            f"fault.passes={passes}",
            "env.task_ids=[2]",
            f"env.init_states_per_task={N_ENVS}",
            "env.init_state_offset=30",
        ],
        env=PromptEnv(),
    )
    harness.model, harness.capture = TokenModel(), None
    harness.setup = lambda: None
    harness.teardown = lambda: None
    harness.run()
    return harness.out_dir


def test_the_collection_logs_what_was_sent_executed_and_targeted(tmp_path):
    run = _collect_run(tmp_path)
    calls = pd.read_parquet(run / "calls.parquet")
    episodes = pd.DataFrame(json.loads((run / "manifest.json").read_text())["episodes"])
    assert len(episodes) == 3 * N_ENVS
    assert set(calls["reset_state_id"]) == set(100 + 30 + np.arange(N_ENVS))
    assert set(episodes["pass_index"]) == {0, 1, 2}
    assert set(episodes["task_description"]) == {"put the bowl on the plate"}
    assert {"clean"} < set(episodes["fault"])  # clean and faulted episodes
    for _, row in calls.iterrows():
        fault = faults.parse(row["fault"])
        cmd = np.asarray(row["a_cmd_env"]).reshape(CHUNKS, DIM)
        sent = np.asarray(row["a_sent"]).reshape(CHUNKS, DIM)
        np.testing.assert_allclose(
            np.asarray(row["a_target"]).reshape(CHUNKS, DIM), fault.compensate(cmd)
        )
        np.testing.assert_allclose(
            sent, fault.compensate(cmd) if row["compensated"] else cmd
        )
        np.testing.assert_allclose(
            np.asarray(row["a_exec"]).reshape(CHUNKS, DIM), fault.apply(sent)
        )
    loaded = ec_dataset.load_calls([run])
    assert ec_dataset.round_trip_error(loaded, TOKENIZER) < 1e-6
    assert all(Path(p).is_file() for p in loaded["frame"])
    counts = ec_dataset.family_counts(loaded)
    assert sum(counts.values()) == 3 * N_ENVS
    summary = report.collection(run, expected_episodes=3 * N_ENVS)
    assert summary["logged_gate"] and summary["episodes"] == 3 * N_ENVS
    assert sum(g["n"] for g in summary["by_group"].values()) == 3 * N_ENVS


def test_masked_targets_keep_the_policy_tokens_before_the_history_has_content(tmp_path):
    """Amendment 4: call 0 trains on the policy's own tokens; later calls do not change."""
    calls = ec_dataset.load_calls([_collect_run(tmp_path)])
    stats = ec_dataset.fit_input_stats(calls)
    plain = ec_dataset.build_samples(calls, "copy", stats, TOKENIZER)
    masked = ec_dataset.build_samples(
        calls, "copy", stats, TOKENIZER, own_target_calls=1
    )
    first = calls["call_idx"].to_numpy() == 0
    own = np.stack([np.asarray(t).reshape(CHUNKS, DIM) for t in calls["action_tokens"]])
    np.testing.assert_array_equal(masked.targets[first], own[first])
    np.testing.assert_array_equal(masked.targets[~first], plain.targets[~first])
    faulted = first & (calls["fault"] != "clean").to_numpy()
    assert faulted.any() and (plain.targets[faulted] != own[faulted]).any()
    np.testing.assert_array_equal(masked.efference, plain.efference)


# -- training and evaluation -----------------------------------------------------


class FakeProcessor:
    """Prompt and image to ids and pixels, deterministically."""

    def __init__(self):
        self.texts = []

    def __call__(self, text, images, proprio_states, padding, max_length):
        self.texts += list(text)
        ids = [[1, 5 + len(t) % 30, 6 + sum(map(ord, t)) % 30, 7, 3] for t in text]
        pixels = torch.as_tensor(images["images"]).float()  # [B, 1, C, H, W]
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.ones(len(ids), 5, dtype=torch.long),
            "pixel_values": pixels.mean(dim=(-1, -2), keepdim=True) / 255.0,
        }


class TrainFakeOFT(fake_oft.FakeOFT):
    """The fake OFT at LIBERO's chunk shape, with the attributes training reads."""

    def __init__(self):
        torch.manual_seed(0)
        super().__init__(action_dim=DIM, chunks=CHUNKS)
        self.vocab_size, self.bin_centers = VOCAB, CENTERS
        self.norm_stats = {}
        self.max_prompt_length = 8
        self.input_processor = FakeProcessor()
        self.efference_inputs = []

    def get_action_stats(self, unnorm_key):
        return {"q01": [-1.0] * 7, "q99": [1.0] * 7, "mask": MASK}

    @torch.no_grad()
    def predict_action_batch(
        self, input_ids=None, attention_mask=None, pixel_values=None, env_obs=None, **kw
    ):
        if env_obs is not None and "main_images" in env_obs:
            if "efference" in env_obs:
                self.efference_inputs.append(np.asarray(env_obs["efference"]).copy())
            prompts = [
                f"In: What action should the robot take to {t.lower()}?\nOut: "
                for t in env_obs["task_descriptions"]
            ]
            images = torch.as_tensor(env_obs["main_images"]).unsqueeze(1)
            inputs = self.input_processor(
                text=prompts,
                images={"images": images.permute(0, 1, 4, 2, 3)},
                proprio_states=env_obs["states"],
                padding="max_length",
                max_length=self.max_prompt_length,
            )
            input_ids = inputs["input_ids"].to(torch.long)
            attention_mask = inputs["attention_mask"].to(torch.bool)
            b, n, c, h, w = inputs["pixel_values"].shape
            pixel_values = inputs["pixel_values"].reshape(b, n * c, h, w)
        _, result = super().predict_action_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        tokens = result["forward_inputs"]["action_tokens"].reshape(-1, CHUNKS, DIM)
        result["forward_inputs"]["action_tokens"] = tokens
        actions = TOKENIZER.values(tokens.numpy()).astype(np.float32)
        actions[..., 6] = 1.0
        return torch.from_numpy(actions), result


def _train(tmp_path: Path, run: Path, arm: str, extra=()) -> tuple[Path, dict]:
    out = tmp_path / f"arm_{arm}"
    args = train_ec.parse_args(
        [
            "--config", str(CONFIG), "--runs", str(run), "--arm", arm,
            "--out", str(out), "--batch", "4", "--micro-batch", "2",
            "--epochs", "2", "--warmup", "2", "--log-every", "1000", *extra,
        ]
    )  # fmt: skip
    return out, train_ec.train(args, load_model=lambda cfg: TrainFakeOFT())


@pytest.mark.parametrize("arm", ["A", "C", "D"])
def test_an_arm_trains_saves_and_runs_in_the_harness(tmp_path, arm):
    pytest.importorskip("peft")
    run = _collect_run(tmp_path / "collect")
    out, meta = _train(tmp_path, run, arm)
    assert meta["in_dim"] == (8 if arm == "D" else 136)
    assert meta["summary"]["steps"] == meta["summary"]["planned_steps"] > 0
    lines = (out / "train_log.jsonl").read_text().splitlines()
    log = [json.loads(line) for line in lines]
    assert len(log) == meta["summary"]["steps"]
    assert all(np.isfinite(entry["nll"]) for entry in log)
    assert all((entry["mse"] is not None) == (arm == "D") for entry in log)
    assert (out / "fm_head.pt").is_file() == (arm == "D")
    projector = torch.load(out / ec_policy.PROJECTOR_FILE)["projector"]
    assert projector["fc2.weight"].abs().sum() > 0  # the token moved off zero
    assert (out / ec_policy.LORA_DIR / "adapter_config.json").is_file()

    model = TrainFakeOFT()
    loaded = ec_policy.load_arm(model, out)
    harness = fake._harness(
        tmp_path / "eval",
        [
            "run_id=eval",
            "hijack.enabled=false",
            "capture.enabled=false",
            "fault.enabled=true",
            "fault.sweep=[bias_x:0.3]",
            "efference.enabled=true",
            f"efference.arm_dir={out}",
        ],
        env=PromptEnv(),
    )
    harness.model, harness.capture, harness.arm = model, None, loaded
    harness._fault = faults.parse("bias_x:0.3")  # run() sets it from the sweep
    calls, _ = fake._collect(harness)
    assert (calls["fault"] == "bias_x:0.3").all() and "a_sent" in calls
    # Online inputs == the offline replay the training samples come from.
    calls = calls.sort_values(["episode_id", "call_idx"]).reset_index(drop=True)
    calls["episode"] = calls["episode_id"]
    replay = ec_dataset.replay_history(calls, loaded["kind"], loaded)
    online = np.stack(
        [
            model.efference_inputs[c][s]
            for c, s in zip(calls["call_idx"], calls["env_slot"])
        ]
    )
    np.testing.assert_allclose(online, replay, atol=1e-6)
    assert np.abs(online[calls["call_idx"].to_numpy() >= 1, 8:]).sum() > 0 or arm == "D"


class TemperatureRecordingOFT(TrainFakeOFT):
    """Records the temperature every training forward is taken at."""

    def __init__(self):
        super().__init__()
        self.forward_temperatures = []

    def default_forward(self, forward_inputs=None, **kw):
        self.forward_temperatures.append(kw.get("temperature"))
        return super().default_forward(forward_inputs=forward_inputs, **kw)


@pytest.mark.parametrize(
    "extra, expected", [((), 1.0), (("--loss-temperature", "1.6"), 1.6)]
)
def test_the_loss_is_taken_at_the_requested_temperature(tmp_path, extra, expected):
    """Amendment 3: the NLL's temperature is a flag, recorded in arm.json."""
    pytest.importorskip("peft")
    run = _collect_run(tmp_path / "collect")
    model = TemperatureRecordingOFT()
    args = train_ec.parse_args(
        [
            "--config", str(CONFIG), "--runs", str(run), "--arm", "A",
            "--out", str(tmp_path / "arm"), "--batch", "4", "--micro-batch", "2",
            "--warmup", "1", "--max-steps", "2", "--log-every", "1000", *extra,
        ]
    )  # fmt: skip
    meta = train_ec.train(args, load_model=lambda cfg: model)
    assert model.forward_temperatures and set(model.forward_temperatures) == {expected}
    assert meta["optim"]["loss_temperature"] == expected
    saved = json.loads((tmp_path / "arm" / ec_policy.ARM_FILE).read_text())
    assert saved["optim"]["loss_temperature"] == expected


def test_zeroed_history_keeps_only_the_state(tmp_path):
    pytest.importorskip("peft")
    run = _collect_run(tmp_path / "collect")
    out, _ = _train(tmp_path, run, "A", ["--max-steps", "2"])
    model = TrainFakeOFT()
    harness = fake._harness(
        tmp_path / "eval",
        [
            "run_id=eval0",
            "hijack.enabled=false",
            "capture.enabled=false",
            "efference.enabled=true",
            f"efference.arm_dir={out}",
            "efference.zero_history=true",
        ],
        env=PromptEnv(),
    )
    harness.model, harness.capture = model, None
    harness.arm = ec_policy.load_arm(model, out)
    fake._collect(harness)
    inputs = np.concatenate(model.efference_inputs)
    assert not inputs[:, 8:].any() and inputs[:, :8].any()


# -- training helpers -------------------------------------------------------------


def test_schedule_batches_and_gate():
    factor = train_ec.cosine_lambda(warmup=4, total=20)
    assert factor(0) == pytest.approx(0.25) and factor(3) == pytest.approx(1.0)
    assert factor(19) < 0.05
    plan = list(train_ec.batches(10, 4, 2, np.random.default_rng(0)))
    assert len(plan) == 4 and {epoch for epoch, _ in plan} == {0, 1}
    assert all(len(set(index)) == 4 for _, index in plan)
    down = [{"nll": 3.0 - 0.01 * i} for i in range(200)]
    assert train_ec.gate_summary(down)["passed"]
    assert not train_ec.gate_summary([{"nll": 1.0}] * 200)["passed"]
    assert not train_ec.gate_summary(down + [{"nll": float("nan")}])["passed"]


def test_model_inputs_build_the_prompt_like_predict():
    model = TrainFakeOFT()
    images = np.full((2, 8, 8, 3), 51, np.uint8)
    out = train_ec.model_inputs(
        model, images, ["Open The Drawer", "b"], np.zeros((2, 8))
    )
    assert model.input_processor.texts[0] == (
        "In: What action should the robot take to open the drawer?\nOut: "
    )
    assert out["pixel_values"].shape == (2, 3, 1, 1)
    assert out["attention_mask"].dtype == torch.bool
    np.testing.assert_allclose(out["pixel_values"].numpy(), 0.2, atol=1e-6)


# -- the readout -------------------------------------------------------------------


def _eval_run(root: Path, name: str, cells: dict[str, list[int]]) -> Path:
    run = root / name
    run.mkdir(parents=True)
    episodes, rows = [], []
    for cell, outcomes in cells.items():
        for state, success in enumerate(outcomes):
            episode = len(episodes)
            episodes.append(
                {
                    "episode_id": episode,
                    "task_id": state % 2,
                    "fault": cell,
                    "success": bool(success),
                    "first_success_call": 3 if success else -1,
                }
            )
            rows.append(
                {
                    "episode_id": episode,
                    "call_idx": 0,
                    "reset_state_id": state,
                    "task_id": state % 2,
                    "a_cmd_env": np.zeros(CHUNKS * DIM, np.float32),
                }
            )
    (run / "manifest.json").write_text(json.dumps({"episodes": episodes}))
    pd.DataFrame(rows).to_parquet(run / "calls.parquet")
    return run


def _arms(root: Path, b_bias: int = 0, d_bias: int = 0) -> dict[str, Path]:
    n = 12
    ones, zeros = [1] * n, [0] * n
    faulted = lambda value: {"bias_x:0.3": [value] * n, "invert_x": [value] * n}  # noqa: E731
    return {
        "F": _eval_run(root, "F", {"clean": ones, **faulted(0)}),
        "A": _eval_run(root, "A", {"clean": ones, **faulted(1)}),
        "B": _eval_run(root, "B", faulted(b_bias)),
        "C": _eval_run(root, "C", faulted(0)),
        "D": _eval_run(root, "D", faulted(d_bias)),
        "_zeros": zeros,
    }


def test_rules_read_a_clean_positive_result(tmp_path):
    arms = _arms(tmp_path)
    arms.pop("_zeros")
    out = report.part_b(arms)
    for family in ("bias", "inversion"):
        entry = out["families"][family]
        assert entry["G_f"] == 1.0 and entry["rule_E"] and entry["rule_N"]
        assert entry["rule_L"] and entry["D_minus_B"]["point"] == 0.0
    assert out["rule_0"]["passed"]
    assert out["headline"].startswith("E, N and 0 on both")
    verdict = report.rule_m(arms["F"])
    assert verdict["verdict"] == "fails as predicted"


def test_rules_read_a_fine_tune_effect_and_a_harmful_loss(tmp_path):
    arms = _arms(tmp_path, b_bias=1, d_bias=0)
    arms.pop("_zeros")
    entry = report.part_b(arms)["families"]["bias"]
    assert entry["rule_E"] and not entry["rule_N"] and not entry["rule_L"]
    assert "not the copy" in entry["reading"] and "the loss hurts" in entry["reading"]


def test_rule_m_flags_a_counterexample(tmp_path):
    run = _eval_run(
        tmp_path,
        "pi0",
        {"clean": [1] * 10, "bias_x:0.3": [0] * 10, "invert_x": [1] * 10},
    )
    assert report.rule_m(run)["verdict"] == "counterexample to Rule M"
    weak = _eval_run(
        tmp_path,
        "weak",
        {"clean": [0] * 8 + [1] * 2, "bias_x:0.3": [0] * 10, "invert_x": [0] * 10},
    )
    assert report.rule_m(weak)["verdict"].startswith("descriptive")
