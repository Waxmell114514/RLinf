"""X6 R1: an analytic fault estimator driven by the efference copy.

The frozen policy's command ``a`` is corrected to ``s * (a - b)`` on the x and y
action dims.  The per-axis sign ``s`` and additive bias ``b`` are estimated
online from what was sent and the measured end-effector displacement, with no
learning (PREREG_X6.md, Amendment 6).

Plant model, per call and per axis j in {x, y}, calibrated on clean calls
(a first-order response: the previous displacement carries over):

    ds_j = k_j * sum_t exec_tj + k_last_j * exec_prev_last_j
           + k_psum_j * sum_t exec_prev_tj + alpha_j * ds_prev_j + c_j + noise,
    exec = s * sent + b                                  (the fault model)

so ``ds_j - alpha_j * ds_prev_j - c_j = s_j * u_j + b_j * v_j`` with
``u_j = k_j * sum sent + k_last_j * sent_prev_last + k_psum_j * sum sent_prev``
(the nominal displacement) and ``v_j = chunk * (k_j + k_psum_j) + k_last_j``
(what a unit bias adds; only ``chunk * k_j`` at call 0, with no previous chunk).
That is linear in (s_j, b_j): a two-parameter Bayesian regression, updated
once per call.  A sign flip or a bias is acted on only once its posterior is
confident (``z_sign``, ``z_bias``), which sets the false-alarm rate on clean
episodes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

AXES = (0, 1)


@dataclass(frozen=True)
class EstimatorParams:
    k: tuple[float, float]
    k_last: tuple[float, float]
    k_psum: tuple[float, float]
    alpha: tuple[float, float]
    c: tuple[float, float]
    noise_sd: tuple[float, float]
    prior_sign_sd: float = 0.3
    prior_bias_sd: float = 0.2
    z_sign: float = 3.0
    z_bias: float = 3.0
    chunk: int = 8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EstimatorParams":
        fields = dict(data)
        for key in ("k", "k_last", "k_psum", "alpha", "c", "noise_sd"):
            fields[key] = tuple(float(x) for x in fields[key])
        return cls(**fields)


def calibrate(
    sent: np.ndarray, prev_chunk: np.ndarray, ds_prev: np.ndarray, ds: np.ndarray
) -> dict[str, tuple]:
    """Least squares of the plant model on clean calls.

    sent, prev_chunk: [N, chunk, dim] (prev_chunk zeros at call 0);
    ds_prev, ds: [N, >=2] (ds_prev zeros at call 0).
    Returns k, k_last, k_psum, alpha, c, noise_sd, one value per axis.
    """
    out: dict[str, list] = {
        key: [] for key in ("k", "k_last", "k_psum", "alpha", "c", "noise_sd")
    }
    for j in AXES:
        x = np.column_stack(
            [
                sent[:, :, j].sum(1),
                prev_chunk[:, -1, j],
                prev_chunk[:, :, j].sum(1),
                ds_prev[:, j],
                np.ones(len(sent)),
            ]
        )
        coef, *_ = np.linalg.lstsq(x, ds[:, j], rcond=None)
        resid = ds[:, j] - x @ coef
        for key, value in zip(("k", "k_last", "k_psum", "alpha", "c"), coef):
            out[key].append(float(value))
        out["noise_sd"].append(float(resid.std()))
    return {key: tuple(values) for key, values in out.items()}


class FaultEstimator:
    """Per-slot, per-axis posterior over (sign, bias); one update per call."""

    def __init__(self, num_envs: int, params: EstimatorParams):
        self.p = params
        self.num_envs = num_envs
        self.mean = np.zeros((num_envs, 2, 2))
        self.cov = np.zeros((num_envs, 2, 2, 2))
        self.prev_chunk = np.zeros((num_envs, params.chunk, 2))
        self.ds_prev = np.zeros((num_envs, 2))
        self.has_prev = np.zeros(num_envs, dtype=bool)
        self.reset()

    def reset(self, slots=None) -> None:
        slots = range(self.num_envs) if slots is None else slots
        prior = np.diag([self.p.prior_sign_sd**2, self.p.prior_bias_sd**2])
        for slot in slots:
            self.mean[slot] = [[1.0, 0.0], [1.0, 0.0]]
            self.cov[slot] = prior
            self.prev_chunk[slot] = 0.0
            self.ds_prev[slot] = 0.0
            self.has_prev[slot] = False

    def update(self, sent: np.ndarray, ds: np.ndarray) -> None:
        """sent: [num_envs, chunk, dim] as sent; ds: [num_envs, >=2] measured."""
        sent = np.asarray(sent, dtype=np.float64)
        ds = np.asarray(ds, dtype=np.float64)
        for a, j in enumerate(AXES):
            k, kl, kps = self.p.k[a], self.p.k_last[a], self.p.k_psum[a]
            u = (
                k * sent[:, :, j].sum(1)
                + kl * self.prev_chunk[:, -1, a]
                + kps * self.prev_chunk[:, :, a].sum(1)
            )
            v = self.p.chunk * k + np.where(self.has_prev, self.p.chunk * kps + kl, 0.0)
            y = ds[:, j] - self.p.alpha[a] * self.ds_prev[:, a] - self.p.c[a]
            r = self.p.noise_sd[a] ** 2
            for slot in range(self.num_envs):
                m, cov = self.mean[slot, a], self.cov[slot, a]
                hs = np.array([u[slot], v[slot]])
                s = hs @ cov @ hs + r
                gain = cov @ hs / s
                self.mean[slot, a] = m + gain * (y[slot] - hs @ m)
                self.cov[slot, a] = cov - np.outer(gain, hs) @ cov
        self.prev_chunk = sent[:, :, list(AXES)].copy()
        self.ds_prev = ds[:, list(AXES)].copy()
        self.has_prev[:] = True

    def estimate(self) -> tuple[np.ndarray, np.ndarray]:
        """(sign [N, 2] in {+1, -1}, bias [N, 2]): what the correction applies.

        The sign flips once its posterior is confidently negative.  The bias is
        then read from the posterior conditioned on that sign (b | s = s_hat),
        and applied only when it is ``z_bias`` posterior sds away from zero.
        """
        sign = np.ones((self.num_envs, 2))
        bias = np.zeros((self.num_envs, 2))
        for a in range(2):
            m_s, m_b = self.mean[:, a, 0], self.mean[:, a, 1]
            c_ss = self.cov[:, a, 0, 0]
            c_sb = self.cov[:, a, 0, 1]
            c_bb = self.cov[:, a, 1, 1]
            flipped = (m_s < 0) & (np.abs(m_s - 1.0) / np.sqrt(c_ss) > self.p.z_sign)
            s_hat = np.where(flipped, -1.0, 1.0)
            b_mean = m_b + c_sb / c_ss * (s_hat - m_s)
            b_sd = np.sqrt(np.maximum(c_bb - c_sb**2 / c_ss, 1e-12))
            sign[:, a] = s_hat
            bias[:, a] = np.where(np.abs(b_mean) / b_sd > self.p.z_bias, b_mean, 0.0)
        return sign, bias

    def correct(self, command: np.ndarray, bound: float = 1.0) -> np.ndarray:
        """What to send for ``command`` [N, chunk, dim]: s * (a - b) on x, y."""
        sign, bias = self.estimate()
        out = np.array(command, dtype=np.float32, copy=True)
        for a, j in enumerate(AXES):
            out[:, :, j] = sign[:, None, a] * (out[:, :, j] - bias[:, None, a])
        out[..., :6] = np.clip(out[..., :6], -bound, bound)
        return out


def replay(
    params: EstimatorParams,
    sent: np.ndarray,
    ds: np.ndarray,
    episodes: np.ndarray,
    call_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Open-loop replay over logged calls (sorted by episode, call).

    Returns the (sign, bias) the estimator would apply *at* each call, i.e.
    from the calls before it in the same episode.
    """
    signs = np.ones((len(sent), 2))
    biases = np.zeros((len(sent), 2))
    est: Optional[FaultEstimator] = None
    for i in range(len(sent)):
        if call_idx[i] == 0 or est is None or episodes[i] != episodes[i - 1]:
            est = FaultEstimator(1, params)
        sign, bias = est.estimate()
        signs[i], biases[i] = sign[0], bias[0]
        est.update(sent[i : i + 1], ds[i : i + 1])
    return signs, biases
