"""X6 R1: the analytic fault estimator (fault_estimator.py) and its harness wiring."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("omegaconf")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efference_probe import faults  # noqa: E402
from efference_probe.fault_estimator import (  # noqa: E402
    EstimatorParams,
    FaultEstimator,
    calibrate,
    replay,
)

PARAMS_FILE = (
    Path(__file__).resolve().parents[1] / "configs" / "x6_estimator_params.json"
)
CHUNK, DIM = 8, 7
K, K_LAST, K_PSUM, ALPHA, NOISE = 0.0093, 0.019, -0.0092, 0.8, 0.004


def _episode(rng, fault, calls=30):
    """A synthetic first-order plant on x, y driven by a wandering command."""
    sent_all, ds_all = [], []
    prev_exec, ds_prev = np.zeros((CHUNK, DIM)), np.zeros(DIM)
    target = rng.uniform(-0.5, 0.5, size=2)
    for call in range(calls):
        cmd = np.zeros((CHUNK, DIM), dtype=np.float32)
        cmd[:, :2] = target + 0.3 * rng.standard_normal((CHUNK, 2))
        if call % 6 == 5:
            target = rng.uniform(-0.5, 0.5, size=2)
        exec_ = fault.apply(cmd)
        ds = np.zeros(DIM)
        ds[:2] = (
            K * exec_[:, :2].sum(0)
            + K_LAST * prev_exec[-1, :2]
            + K_PSUM * prev_exec[:, :2].sum(0)
            + ALPHA * ds_prev[:2]
            + NOISE * rng.standard_normal(2)
        )
        sent_all.append(cmd)
        ds_all.append(ds)
        prev_exec, ds_prev = exec_, ds
    return np.stack(sent_all), np.stack(ds_all)


def _params(**kw):
    return EstimatorParams(
        k=(K, K), k_last=(K_LAST, K_LAST), k_psum=(K_PSUM, K_PSUM),
        alpha=(ALPHA, ALPHA), c=(0.0, 0.0), noise_sd=(NOISE, NOISE), **kw,
    )  # fmt: skip


def _run(fault, seed=0):
    sent, ds = _episode(np.random.default_rng(seed), fault)
    return replay(_params(), sent, ds, np.zeros(len(sent)), np.arange(len(sent)))


def test_calibration_recovers_the_plant():
    rng = np.random.default_rng(1)
    sents, dss, prevs, dsps = [], [], [], []
    for _ in range(40):
        sent, ds = _episode(rng, faults.parse("clean"))
        prev = np.concatenate([np.zeros((1, CHUNK, DIM)), sent[:-1]])
        dsp = np.concatenate([np.zeros((1, DIM)), ds[:-1]])
        sents.append(sent), dss.append(ds), prevs.append(prev), dsps.append(dsp)
    plant = calibrate(*(np.concatenate(x) for x in (sents, prevs, dsps, dss)))
    np.testing.assert_allclose(plant["k"], K, rtol=0.05)
    np.testing.assert_allclose(plant["alpha"], ALPHA, atol=0.03)
    np.testing.assert_allclose(plant["noise_sd"], NOISE, rtol=0.1)


def test_clean_stays_uncorrected_and_faults_are_identified():
    signs, biases = _run(faults.parse("clean"))
    assert (signs == 1).all() and (biases == 0).all()
    signs, biases = _run(faults.parse("bias_x:0.3"))
    assert (biases[10:, 0] != 0).all() and abs(biases[-1, 0] - 0.3) < 0.08
    assert abs(biases[-1, 1]) < 0.08 and (signs == 1).all()
    signs, biases = _run(faults.parse("invert_x"))
    assert (signs[10:, 0] == -1).all() and (signs[:, 1] == 1).all()
    assert (signs[0] == 1).all()  # nothing is known before the first call


def test_correction_inverts_the_estimated_fault():
    est = FaultEstimator(1, _params())
    est.mean[0] = [[-1.0, 0.0], [1.0, 0.25]]
    est.cov[0] = np.diag([1e-4, 1e-4])
    cmd = np.full((1, CHUNK, DIM), 0.2, dtype=np.float32)
    out = est.correct(cmd)
    np.testing.assert_allclose(out[0, :, 0], -0.2, atol=1e-6)
    np.testing.assert_allclose(out[0, :, 1], 0.2 - 0.25, atol=1e-6)
    np.testing.assert_allclose(out[0, :, 2:], 0.2)


def test_the_committed_params_load():
    params = EstimatorParams.from_dict(json.loads(PARAMS_FILE.read_text()))
    assert len(params.k) == 2 and params.z_bias > 0


def test_the_harness_sends_the_estimators_correction(tmp_path):
    from efference_probe.tests.test_x6_ec import PromptEnv, TokenModel, fake

    chunks, dim = fake.CHUNKS, fake.DIM

    harness = fake._harness(
        tmp_path,
        [
            "run_id=est",
            "hijack.enabled=false",
            "capture.enabled=false",
            "fault.enabled=true",
            "fault.sweep=[invert_x]",
            "estimator.enabled=true",
            f"estimator.params_file={PARAMS_FILE}",
        ],
        env=PromptEnv(),
    )
    harness.model, harness.capture = TokenModel(), None
    harness.estimator_params = EstimatorParams.from_dict(
        json.loads(PARAMS_FILE.read_text())
    )
    harness._fault = faults.parse("invert_x")
    calls, _ = fake._collect(harness)
    fault = faults.parse("invert_x")
    for _, row in calls.iterrows():
        cmd = np.asarray(row["a_cmd_env"]).reshape(chunks, dim).copy()
        sign, bias = np.asarray(row["est_sign"]), np.asarray(row["est_bias"])
        expected = cmd.copy()
        for a in range(2):
            expected[:, a] = sign[a] * (cmd[:, a] - bias[a])
        expected[:, :6] = np.clip(expected[:, :6], -1, 1)
        assert np.asarray(row["est_sign"]).shape == (2,)
        np.testing.assert_allclose(row["a_sent"], expected.reshape(-1), atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(row["a_exec"]).reshape(chunks, dim),
            fault.apply(expected),
            atol=1e-6,
        )
    for bad in (["fault.oracle=true"], ["estimator.params_file="]):
        with pytest.raises(ValueError):
            fake._harness(
                tmp_path / "bad",
                ["hijack.enabled=false", "capture.enabled=false",
                 "fault.enabled=true", "fault.sweep=[invert_x]",
                 "estimator.enabled=true",
                 f"estimator.params_file={PARAMS_FILE}", *bad],
                env=PromptEnv(),
            )  # fmt: skip


SELECT_FILE = PARAMS_FILE.with_name("x6_estimator_params_select.json")


def _select_run(fault, seed=0):
    from efference_probe.fault_estimator import EstimatorParams as P

    params = P(**{**_params().to_dict(), "kind": "select", "p_act": 0.8})
    sent, ds = _episode(np.random.default_rng(seed), fault)
    return replay(params, sent, ds, np.zeros(len(sent)), np.arange(len(sent)))


def test_hypothesis_selection_identifies_each_fault_without_a_spurious_bias():
    signs, biases = _select_run(faults.parse("clean"))
    assert (signs == 1).all() and (biases == 0).all()
    signs, biases = _select_run(faults.parse("bias_x:0.3"))
    assert (signs == 1).all() and abs(biases[-1, 0] - 0.3) < 0.08
    signs, biases = _select_run(faults.parse("invert_x"))
    assert (signs[5:, 0] == -1).all() and (signs[:, 1] == 1).all()
    # The inversion is never sent as a bias, before or after the flip.
    assert (biases == 0).all()


def test_the_committed_selection_params_load_and_dispatch():
    from efference_probe.fault_estimator import SelectionEstimator, make_estimator

    params = EstimatorParams.from_dict(json.loads(SELECT_FILE.read_text()))
    assert params.kind == "select"
    assert isinstance(make_estimator(3, params), SelectionEstimator)
    with pytest.raises(ValueError):
        make_estimator(1, EstimatorParams(**{**params.to_dict(), "kind": "x"}))
