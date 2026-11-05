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
    # "joint": the (sign, bias) regression above (Amendment 6).  "select":
    # Amendment 7's discrete hypotheses per axis (SelectionEstimator).
    kind: str = "joint"
    # "select" only: prior probabilities of no fault, a bias, an inversion;
    # the bias prior's sd; and the posterior probability a hypothesis needs
    # before its correction is applied.
    prior_h: tuple[float, float, float] = (0.6, 0.2, 0.2)
    bias_sd: float = 0.2
    p_act: float = 0.95
    # "affine" only (Amendment 8, AffineEstimator): the prior sd of each entry
    # of the 2 x 2 map A (prior A = I); the posterior Mahalanobis distance
    # from (I, 0), squared over the 6 parameters, a correction needs; the
    # largest posterior sd of an entry of A at which it may be applied (the
    # excitation gate); and the smallest |det A| ever inverted.
    a_sd: float = 1.0
    chi2_act: float = 25.0
    a_sd_max: float = 0.3
    min_det: float = 0.3
    # "similarity" only: the bias part's chi2 (2 dof) before it is applied.
    chi2_bias: float = 9.0
    # "frame_select" only (Amendment 9, FrameSelectEstimator): prior
    # probabilities of the classes none / translate / rotate / reflect, and the
    # posterior another class needs to replace the latched one.
    prior_frame: tuple[float, float, float, float] = (0.7, 0.1, 0.1, 0.1)
    p_switch: float = 0.99

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EstimatorParams":
        fields = dict(data)
        for key in (
            "k",
            "k_last",
            "k_psum",
            "alpha",
            "c",
            "noise_sd",
            "prior_h",
            "prior_frame",
        ):
            if key not in fields:
                continue
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


class SelectionEstimator:
    """Amendment 7: which fault is active, per axis, as discrete hypotheses.

    H0 no fault (s = +1, b = 0); H1 a bias (s = +1, b ~ N(0, bias_sd^2),
    tracked by a scalar Kalman filter); H2 an inversion (s = -1, b = 0).  Each
    hypothesis's marginal likelihood of the measured displacements updates its
    posterior probability (MOSAIC / COIN-style context inference).  A
    correction is applied only for a hypothesis whose posterior exceeds
    ``p_act``: an implausibly large implied bias, which an inversion produces
    while the command is nearly constant, loses to H2 instead of being sent.
    """

    def __init__(self, num_envs: int, params: EstimatorParams):
        self.p = params
        self.num_envs = num_envs
        self.log_post = np.zeros((num_envs, 2, 3))
        self.b_mean = np.zeros((num_envs, 2))
        self.b_var = np.zeros((num_envs, 2))
        self.prev_chunk = np.zeros((num_envs, params.chunk, 2))
        self.ds_prev = np.zeros((num_envs, 2))
        self.has_prev = np.zeros(num_envs, dtype=bool)
        self.reset()

    def reset(self, slots=None) -> None:
        slots = range(self.num_envs) if slots is None else slots
        for slot in slots:
            self.log_post[slot] = np.log(np.asarray(self.p.prior_h, dtype=float))
            self.b_mean[slot] = 0.0
            self.b_var[slot] = self.p.bias_sd**2
            self.prev_chunk[slot] = 0.0
            self.ds_prev[slot] = 0.0
            self.has_prev[slot] = False

    def update(self, sent: np.ndarray, ds: np.ndarray) -> None:
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

            def loglik(resid, var):
                return -0.5 * (np.log(2 * np.pi * var) + resid**2 / var)

            l0 = loglik(y - u, r)
            l2 = loglik(y + u, r)
            pred_var = v**2 * self.b_var[:, a] + r
            resid = y - u - v * self.b_mean[:, a]
            l1 = loglik(resid, pred_var)
            gain = self.b_var[:, a] * v / pred_var
            self.b_mean[:, a] += gain * resid
            self.b_var[:, a] *= 1.0 - gain * v
            lp = self.log_post[:, a] + np.stack([l0, l1, l2], axis=1)
            self.log_post[:, a] = lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)
        self.prev_chunk = sent[:, :, list(AXES)].copy()
        self.ds_prev = ds[:, list(AXES)].copy()
        self.has_prev[:] = True

    def posterior(self) -> np.ndarray:
        """[num_envs, 2 axes, 3 hypotheses] posterior probabilities."""
        return np.exp(self.log_post)

    def estimate(self) -> tuple[np.ndarray, np.ndarray]:
        post = self.posterior()
        sign = np.where(post[:, :, 2] > self.p.p_act, -1.0, 1.0)
        bias = np.where(post[:, :, 1] > self.p.p_act, self.b_mean, 0.0)
        return sign, bias

    correct = FaultEstimator.correct


class AffineEstimator:
    """Amendment 8: a general affine action-space map, exec = A sent + b on x, y.

    Nothing about the fault families is built in.  Per measured axis j the
    plant model of :class:`FaultEstimator` becomes linear in the 3 parameters
    (A_j0, A_j1, b_j), which a Bayesian regression tracks with prior
    A = I (sd ``a_sd`` per entry), b = 0 (sd ``bias_sd``).  The correction
    A^-1 (a - b) is sent only when (A, b) is ``chi2_act`` away from (I, 0) in
    posterior Mahalanobis distance and every entry of A is pinned down to
    ``a_sd_max`` (so a rotation, an inversion or a bias is acted on only once
    the commands have excited the map enough to tell them apart).
    """

    def __init__(self, num_envs: int, params: EstimatorParams):
        self.p = params
        self.num_envs = num_envs
        self.mean = np.zeros((num_envs, 2, 3))
        self.cov = np.zeros((num_envs, 2, 3, 3))
        self.prev_chunk = np.zeros((num_envs, params.chunk, 2))
        self.ds_prev = np.zeros((num_envs, 2))
        self.has_prev = np.zeros(num_envs, dtype=bool)
        self.reset()

    def prior_mean(self) -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def reset(self, slots=None) -> None:
        slots = range(self.num_envs) if slots is None else slots
        prior = np.diag([self.p.a_sd**2, self.p.a_sd**2, self.p.bias_sd**2])
        for slot in slots:
            self.mean[slot] = self.prior_mean()
            self.cov[slot] = prior
            self.prev_chunk[slot] = 0.0
            self.ds_prev[slot] = 0.0
            self.has_prev[slot] = False

    def update(self, sent: np.ndarray, ds: np.ndarray) -> None:
        sent = np.asarray(sent, dtype=np.float64)
        ds = np.asarray(ds, dtype=np.float64)
        xy = sent[:, :, list(AXES)]
        for a, j in enumerate(AXES):
            k, kl, kps = self.p.k[a], self.p.k_last[a], self.p.k_psum[a]
            # The nominal displacement each input axis i would cause on axis j.
            u = (
                k * xy.sum(1)
                + kl * self.prev_chunk[:, -1, :]
                + kps * self.prev_chunk.sum(1)
            )  # [N, 2]
            v = self.p.chunk * k + np.where(self.has_prev, self.p.chunk * kps + kl, 0.0)
            y = ds[:, j] - self.p.alpha[a] * self.ds_prev[:, a] - self.p.c[a]
            r = self.p.noise_sd[a] ** 2
            h = np.concatenate([u, v[:, None]], axis=1)  # [N, 3]
            for slot in range(self.num_envs):
                m, cov, hs = self.mean[slot, a], self.cov[slot, a], h[slot]
                gain = cov @ hs / (hs @ cov @ hs + r)
                self.mean[slot, a] = m + gain * (y[slot] - hs @ m)
                self.cov[slot, a] = cov - np.outer(gain, hs) @ cov
        self.prev_chunk = xy.copy()
        self.ds_prev = ds[:, list(AXES)].copy()
        self.has_prev[:] = True

    def estimate(self) -> tuple[np.ndarray, np.ndarray]:
        """(A [N, 2, 2], b [N, 2]) the correction applies; (I, 0) when gated."""
        eye = np.broadcast_to(np.eye(2), (self.num_envs, 2, 2)).copy()
        bias = np.zeros((self.num_envs, 2))
        delta = self.mean - self.prior_mean()
        for slot in range(self.num_envs):
            d2 = sum(
                delta[slot, a] @ np.linalg.solve(self.cov[slot, a], delta[slot, a])
                for a in range(2)
            )
            a_sd = np.sqrt(
                np.concatenate([np.diag(self.cov[slot, a])[:2] for a in range(2)])
            )
            amap = self.mean[slot, :, :2]
            if (
                d2 > self.p.chi2_act
                and a_sd.max() <= self.p.a_sd_max
                and abs(np.linalg.det(amap)) >= self.p.min_det
            ):
                eye[slot] = amap
                bias[slot] = self.mean[slot, :, 2]
        return eye, bias

    def correct(self, command: np.ndarray, bound: float = 1.0) -> np.ndarray:
        """What to send for ``command`` [N, chunk, dim]: A^-1 (a - b) on x, y."""
        amap, bias = self.estimate()
        out = np.array(command, dtype=np.float32, copy=True)
        for slot in range(self.num_envs):
            target = out[slot, :, :2].astype(np.float64) - bias[slot]
            out[slot, :, :2] = np.linalg.solve(amap[slot], target.T).T
        out[..., :6] = np.clip(out[..., :6], -bound, bound)
        return out

    @staticmethod
    def row_fields(estimate, slot: int) -> dict[str, np.ndarray]:
        return {
            "est_A": np.asarray(estimate[0][slot], np.float32).reshape(-1),
            "est_bias": np.asarray(estimate[1][slot], np.float32),
        }


class SimilarityEstimator(AffineEstimator):
    """Amendment 8b: a planar frame misalignment, exec_xy = A sent_xy + b.

    A is a scaled rotation [[p, -q], [q, p]] (proper: rotation, gain) or a
    scaled reflection [[p, q], [q, -p]] (improper: a flipped axis convention).
    Each handedness keeps a joint 4-parameter Gaussian posterior over
    (p, q, b_x, b_y), fed by both axes' displacements, and its marginal
    likelihood weighs the two (prior probability ``1 - prior_h[2]`` proper).
    The improper map is applied once its posterior exceeds ``p_act``; the
    proper map once it is ``chi2_act`` (4 dof) from the identity.  Both need
    the map's entries pinned down to ``a_sd_max``.
    """

    def __init__(self, num_envs: int, params: EstimatorParams):
        self.p = params
        self.num_envs = num_envs
        self.mean = np.zeros((num_envs, 2, 4))
        self.cov = np.zeros((num_envs, 2, 4, 4))
        self.log_post = np.zeros((num_envs, 2))
        self.prev_chunk = np.zeros((num_envs, params.chunk, 2))
        self.ds_prev = np.zeros((num_envs, 2))
        self.has_prev = np.zeros(num_envs, dtype=bool)
        self.reset()

    def reset(self, slots=None) -> None:
        slots = range(self.num_envs) if slots is None else slots
        a2, b2 = self.p.a_sd**2, self.p.bias_sd**2
        p_improper = self.p.prior_h[2]
        for slot in slots:
            self.mean[slot, 0] = [1.0, 0.0, 0.0, 0.0]
            self.mean[slot, 1] = [0.0, 0.0, 0.0, 0.0]
            self.cov[slot, 0] = np.diag([a2, a2, b2, b2])
            self.cov[slot, 1] = np.diag([1.0, 1.0, b2, b2])
            self.log_post[slot] = np.log([1.0 - p_improper, p_improper])
            self.prev_chunk[slot] = 0.0
            self.ds_prev[slot] = 0.0
            self.has_prev[slot] = False

    def _features(self, xy: np.ndarray):
        """Per axis j: u[:, i] (input axis i's nominal effect on j), v, y-part."""
        out = []
        for a in range(2):
            k, kl, kps = self.p.k[a], self.p.k_last[a], self.p.k_psum[a]
            u = (
                k * xy.sum(1)
                + kl * self.prev_chunk[:, -1, :]
                + kps * self.prev_chunk.sum(1)
            )
            v = self.p.chunk * k + np.where(self.has_prev, self.p.chunk * kps + kl, 0.0)
            out.append((u, v))
        return out

    def update(self, sent: np.ndarray, ds: np.ndarray) -> None:
        sent = np.asarray(sent, dtype=np.float64)
        ds = np.asarray(ds, dtype=np.float64)
        xy = sent[:, :, list(AXES)]
        (ux, vx), (uy, vy) = self._features(xy)
        y = np.stack(
            [
                ds[:, j] - self.p.alpha[a] * self.ds_prev[:, a] - self.p.c[a]
                for a, j in enumerate(AXES)
            ],
            axis=1,
        )
        r = np.diag([self.p.noise_sd[0] ** 2, self.p.noise_sd[1] ** 2])
        for slot in range(self.num_envs):
            zeros = 0.0
            h_plus = np.array(
                [
                    [ux[slot, 0], -ux[slot, 1], vx[slot], zeros],
                    [uy[slot, 1], uy[slot, 0], zeros, vy[slot]],
                ]
            )
            h_minus = np.array(
                [
                    [ux[slot, 0], ux[slot, 1], vx[slot], zeros],
                    [-uy[slot, 1], uy[slot, 0], zeros, vy[slot]],
                ]
            )
            for hyp, h in enumerate((h_plus, h_minus)):
                m, cov = self.mean[slot, hyp], self.cov[slot, hyp]
                s = h @ cov @ h.T + r
                resid = y[slot] - h @ m
                sign, logdet = np.linalg.slogdet(2 * np.pi * s)
                self.log_post[slot, hyp] += -0.5 * (
                    logdet + resid @ np.linalg.solve(s, resid)
                )
                gain = cov @ h.T @ np.linalg.inv(s)
                self.mean[slot, hyp] = m + gain @ resid
                self.cov[slot, hyp] = cov - gain @ h @ cov
            self.log_post[slot] -= np.logaddexp.reduce(self.log_post[slot])
        self.prev_chunk = xy.copy()
        self.ds_prev = ds[:, list(AXES)].copy()
        self.has_prev[:] = True

    @staticmethod
    def _map(theta: np.ndarray, improper: bool) -> np.ndarray:
        p, q = theta[0], theta[1]
        if improper:
            return np.array([[p, q], [q, -p]])
        return np.array([[p, -q], [q, p]])

    def estimate(self) -> tuple[np.ndarray, np.ndarray]:
        """(A, b) to invert.  Scale is a nuisance: the map is sent at unit scale.

        Improper (a flipped axis): applied once its posterior exceeds p_act.
        Proper: the rotation part (q) is applied once its chi2 (1 dof) passes
        ``chi2_act``; the bias once its chi2 (2 dof) passes ``chi2_bias``.
        Clean episodes vary mostly in scale (p), which is never acted on.
        """
        eye = np.broadcast_to(np.eye(2), (self.num_envs, 2, 2)).copy()
        bias = np.zeros((self.num_envs, 2))
        for slot in range(self.num_envs):
            post = np.exp(self.log_post[slot])
            improper = post[1] > self.p.p_act
            hyp = 1 if improper else 0
            m, cov = self.mean[slot, hyp], self.cov[slot, hyp]
            if np.sqrt(np.diag(cov)[:2]).max() > self.p.a_sd_max:
                continue
            amap = self._map(m, improper=improper)
            scale = np.sqrt(abs(np.linalg.det(amap)))
            if scale < np.sqrt(self.p.min_det):
                continue
            rotate = improper or m[1] ** 2 / cov[1, 1] > self.p.chi2_act
            if rotate:
                eye[slot] = amap / scale
            b = m[2:]
            if b @ np.linalg.solve(cov[2:, 2:], b) > self.p.chi2_bias:
                bias[slot] = b
        return eye, bias


FRAME_CLASSES = ("none", "translate", "rotate", "reflect")


class FrameSelectEstimator(AffineEstimator):
    """Amendment 9: which *structural class* of planar map is active.

    Classes (exec_xy = A sent_xy + b): none (A = I, b = 0); translate (A = I,
    b ~ N(0, bias_sd^2 I)); rotate (A = [[p, -q], [q, p]], b = 0, (p, q) ~
    N((1, 0), a_sd^2)); reflect (A = [[p, q], [q, -p]], b = 0, (p, q) ~
    N(0, I)).  Each class is a linear-Gaussian model whose marginal likelihood
    updates its posterior (``prior_frame``), so a translation and a frame
    change compete instead of one absorbing the other's residual (the
    Amendment 8 failure).  Nothing names a fault family or angle.

    The applied class latches per episode: it is first adopted when its
    posterior exceeds ``p_act`` and is replaced only when another non-"none"
    class exceeds ``p_switch``.  Its parameters keep updating.  A rotate map
    is applied at unit scale, and only once its rotation part q is
    ``chi2_act`` (1 dof) from zero (scale alone is a nuisance); maps also need
    their entries' sd below ``a_sd_max``.
    """

    def __init__(self, num_envs: int, params: EstimatorParams):
        self.p = params
        self.num_envs = num_envs
        n = len(FRAME_CLASSES)
        self.mean = np.zeros((num_envs, n, 2))
        self.cov = np.zeros((num_envs, n, 2, 2))
        self.log_post = np.zeros((num_envs, n))
        self.active = np.zeros(num_envs, dtype=int)
        self.prev_chunk = np.zeros((num_envs, params.chunk, 2))
        self.ds_prev = np.zeros((num_envs, 2))
        self.has_prev = np.zeros(num_envs, dtype=bool)
        self.reset()

    def reset(self, slots=None) -> None:
        slots = range(self.num_envs) if slots is None else slots
        prior = np.log(np.asarray(self.p.prior_frame, dtype=float))
        for slot in slots:
            self.mean[slot] = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
            self.cov[slot, 0] = np.eye(2)  # unused (no parameters)
            self.cov[slot, 1] = np.eye(2) * self.p.bias_sd**2
            self.cov[slot, 2] = np.eye(2) * self.p.a_sd**2
            self.cov[slot, 3] = np.eye(2)
            self.log_post[slot] = prior
            self.active[slot] = 0
            self.prev_chunk[slot] = 0.0
            self.ds_prev[slot] = 0.0
            self.has_prev[slot] = False

    _features = SimilarityEstimator._features

    def update(self, sent: np.ndarray, ds: np.ndarray) -> None:
        sent = np.asarray(sent, dtype=np.float64)
        ds = np.asarray(ds, dtype=np.float64)
        xy = sent[:, :, list(AXES)]
        (ux, vx), (uy, vy) = self._features(xy)
        y = np.stack(
            [
                ds[:, j] - self.p.alpha[a] * self.ds_prev[:, a] - self.p.c[a]
                for a, j in enumerate(AXES)
            ],
            axis=1,
        )
        r = np.diag([self.p.noise_sd[0] ** 2, self.p.noise_sd[1] ** 2])
        for slot in range(self.num_envs):
            nominal = np.array([ux[slot, 0], uy[slot, 1]])
            # (offset, design) per class: y = offset + H theta + noise.
            models = (
                (nominal, None),
                (nominal, np.diag([vx[slot], vy[slot]])),
                (
                    np.zeros(2),
                    np.array([[ux[slot, 0], -ux[slot, 1]], [uy[slot, 1], uy[slot, 0]]]),
                ),
                (
                    np.zeros(2),
                    np.array([[ux[slot, 0], ux[slot, 1]], [-uy[slot, 1], uy[slot, 0]]]),
                ),
            )
            for cls, (offset, h) in enumerate(models):
                if h is None:
                    s, resid = r, y[slot] - offset
                else:
                    m, cov = self.mean[slot, cls], self.cov[slot, cls]
                    s = h @ cov @ h.T + r
                    resid = y[slot] - offset - h @ m
                    gain = cov @ h.T @ np.linalg.inv(s)
                    self.mean[slot, cls] = m + gain @ resid
                    self.cov[slot, cls] = cov - gain @ h @ cov
                _, logdet = np.linalg.slogdet(2 * np.pi * s)
                self.log_post[slot, cls] += -0.5 * (
                    logdet + resid @ np.linalg.solve(s, resid)
                )
            self.log_post[slot] -= np.logaddexp.reduce(self.log_post[slot])
            post = np.exp(self.log_post[slot])
            best = int(np.argmax(post[1:])) + 1
            if self.active[slot] == 0:
                if post[best] > self.p.p_act:
                    self.active[slot] = best
            elif best != self.active[slot] and post[best] > self.p.p_switch:
                self.active[slot] = best
        self.prev_chunk = xy.copy()
        self.ds_prev = ds[:, list(AXES)].copy()
        self.has_prev[:] = True

    def estimate(self) -> tuple[np.ndarray, np.ndarray]:
        """(A, b) of the latched class; (I, 0) while none is or it is gated."""
        eye = np.broadcast_to(np.eye(2), (self.num_envs, 2, 2)).copy()
        bias = np.zeros((self.num_envs, 2))
        for slot in range(self.num_envs):
            cls = self.active[slot]
            m, cov = self.mean[slot, cls], self.cov[slot, cls]
            if cls == 1:
                bias[slot] = m
                continue
            if cls == 0 or np.sqrt(np.diag(cov)).max() > self.p.a_sd_max:
                continue
            if cls == 2 and m[1] ** 2 / cov[1, 1] <= self.p.chi2_act:
                continue
            amap = SimilarityEstimator._map(m, improper=cls == 3)
            scale = np.sqrt(abs(np.linalg.det(amap)))
            if scale < np.sqrt(self.p.min_det):
                continue
            eye[slot] = amap / scale
        return eye, bias


def _sign_bias_fields(estimate, slot: int) -> dict[str, np.ndarray]:
    return {
        "est_sign": np.asarray(estimate[0][slot], np.float32),
        "est_bias": np.asarray(estimate[1][slot], np.float32),
    }


FaultEstimator.row_fields = staticmethod(_sign_bias_fields)
SelectionEstimator.row_fields = staticmethod(_sign_bias_fields)


def make_estimator(num_envs: int, params: EstimatorParams):
    """The estimator ``params.kind`` names."""
    if params.kind == "affine":
        return AffineEstimator(num_envs, params)
    if params.kind == "similarity":
        return SimilarityEstimator(num_envs, params)
    if params.kind == "frame_select":
        return FrameSelectEstimator(num_envs, params)
    if params.kind == "select":
        return SelectionEstimator(num_envs, params)
    if params.kind == "joint":
        return FaultEstimator(num_envs, params)
    raise ValueError(f"unknown estimator kind {params.kind!r}")


def replay(
    params: EstimatorParams,
    sent: np.ndarray,
    ds: np.ndarray,
    episodes: np.ndarray,
    call_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Open-loop replay over logged calls (sorted by episode, call).

    Returns what the estimator would apply *at* each call, from the calls
    before it in the same episode: (sign, bias), or (A, bias) for "affine".
    """
    firsts, biases = [], []
    est: Optional[Any] = None
    for i in range(len(sent)):
        if call_idx[i] == 0 or est is None or episodes[i] != episodes[i - 1]:
            est = make_estimator(1, params)
        first, bias = est.estimate()
        firsts.append(first[0])
        biases.append(bias[0])
        est.update(sent[i : i + 1], ds[i : i + 1])
    return np.asarray(firsts), np.asarray(biases)
