"""Regime layer (spec 2.2, 7.5, 16.6): package-independent HMM parameters, alignment, an owned forward-only filter,
the EM -> export -> align -> filter -> boundary -> OOS pipeline, and the fixed fixture regime table.

Investable probabilities come only from the owned scaled log-space forward recursion ``p(z_t | y_1..y_t)``. There is no
backward pass and no smoothing; ``hmmlearn.predict_proba`` is never used.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import logsumexp

LOG_2PI = float(np.log(2.0 * np.pi))
VARIANCE_FLOOR = 1e-6  # covariance floor applied by validation (spec 7.5 step 3)


# ---------------------------------------------------------------------------------------- likelihood-level kernel
def _check(p: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError(f"{name} must be finite and nonnegative")


def predict(prior: np.ndarray, transition: np.ndarray) -> np.ndarray:
    prior = np.asarray(prior, dtype=float)
    transition = np.asarray(transition, dtype=float)
    _check(prior, "prior")
    _check(transition, "transition")
    if transition.shape != (prior.size, prior.size) or not np.allclose(transition.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("transition must be a row-stochastic square matrix matching the state count")
    return prior @ transition


def update(predicted: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    likelihood = np.asarray(likelihood, dtype=float)
    _check(likelihood, "likelihood")
    joint = np.asarray(predicted, dtype=float) * likelihood
    total = joint.sum()
    if total <= 0:
        raise ValueError("zero-probability observation under the model")
    return joint / total


def filter_step(prior: np.ndarray, transition: np.ndarray, likelihood: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted, filtered) for one observation."""
    pred = predict(prior, transition)
    return pred, update(pred, likelihood)


def filter_window(start: np.ndarray, transition: np.ndarray, likelihoods: np.ndarray) -> np.ndarray:
    """Filter a window of likelihood rows; the first row's predicted prior is the fitted start distribution."""
    likelihoods = np.asarray(likelihoods, dtype=float)
    out = np.empty_like(likelihoods)
    predicted = np.asarray(start, dtype=float)
    for i, lik in enumerate(likelihoods):
        predicted = predicted if i == 0 else predict(out[i - 1], transition)
        out[i] = update(predicted, lik)
    return out


# ---------------------------------------------------------------------------------------- parameter schema
@dataclass(frozen=True)
class HMMParameters:
    """Package-independent diagonal-Gaussian HMM parameters (no hmmlearn objects leak past export)."""

    start: tuple[float, ...]
    transition: tuple[tuple[float, ...], ...]
    means: tuple[tuple[float, ...], ...]
    variances: tuple[tuple[float, ...], ...]
    feature_order: tuple[str, ...]
    converged: bool
    fitter: str

    @property
    def n_states(self) -> int:
        return len(self.start)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> HMMParameters:
        d = json.loads(text)
        return cls(tuple(d["start"]), tuple(map(tuple, d["transition"])), tuple(map(tuple, d["means"])),
                   tuple(map(tuple, d["variances"])), tuple(d["feature_order"]), bool(d["converged"]), d["fitter"])

    def arrays(self) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        return (np.array(self.start), np.array(self.transition), np.array(self.means), np.array(self.variances))


def validate_parameters(p: HMMParameters, n_features: int | None = None) -> None:
    s, a, m, v = p.arrays()
    k = p.n_states
    if a.shape != (k, k) or m.shape[0] != k or v.shape != m.shape:
        raise ValueError("inconsistent HMM parameter shapes")
    if n_features is not None and m.shape[1] != n_features:
        raise ValueError("feature count mismatch")
    if len(p.feature_order) != m.shape[1]:
        raise ValueError("feature_order length mismatch")
    del s  # an invalid fitted start vector is handled by the reported stationary fallback in the pipeline (spec 7.5)
    for name, arr in (("transition", a), ("means", m), ("variances", v)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
    if np.any(a < 0) or not np.allclose(a.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("transition rows must be stochastic")
    if np.any(v < VARIANCE_FLOOR * 0.999):
        raise ValueError("variance below floor")


def state_volatility(p: HMMParameters) -> NDArray[np.float64]:
    """Fitted state volatility: root of the mean per-feature variance."""
    return np.sqrt(np.array(p.variances).mean(axis=1))


def align_states(p: HMMParameters) -> tuple[HMMParameters, tuple[int, ...]]:
    """Order by ascending fitted volatility, then mean, then original index; one permutation for every parameter."""
    vol, mean = state_volatility(p), np.array(p.means).mean(axis=1)
    perm = tuple(sorted(range(p.n_states), key=lambda i: (vol[i], mean[i], i)))
    idx = list(perm)
    a = np.array(p.transition)[np.ix_(idx, idx)]
    return HMMParameters(
        tuple(np.array(p.start)[idx]), tuple(map(tuple, a)), tuple(map(tuple, np.array(p.means)[idx])),
        tuple(map(tuple, np.array(p.variances)[idx])), p.feature_order, p.converged, p.fitter), perm


# ---------------------------------------------------------------------------------------- owned forward filter
def log_emissions(p: HMMParameters, obs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Diagonal-Gaussian log-likelihoods, shape (rows, states)."""
    _, _, m, v = p.arrays()
    y = np.asarray(obs, dtype=float)
    if y.ndim != 2 or y.shape[1] != m.shape[1]:
        raise ValueError("observation matrix does not match the parameter feature order")
    d = y[:, None, :] - m[None, :, :]
    return -0.5 * np.sum(d * d / v[None] + np.log(v[None]) + LOG_2PI, axis=2)


@dataclass(frozen=True)
class ForwardResult:
    predicted: NDArray[np.float64]
    filtered: NDArray[np.float64]
    log_scale: NDArray[np.float64]  # log p(y_t | y_<t): the scaling constant per row


def forward_filter(p: HMMParameters, obs: NDArray[np.float64], prior: NDArray[np.float64] | None = None,
                   *, first_is_start: bool = True) -> ForwardResult:
    """Scaled log-space forward recursion.

    With ``first_is_start`` the first row's predicted prior is the fitted start vector (train window). When continuing
    from a boundary posterior pass ``prior=boundary`` and ``first_is_start=False``: the first row is predicted through
    the transition matrix, never reset to stationarity.
    """
    s, a, _, _ = p.arrays()
    ll = log_emissions(p, obs)
    log_a = np.log(np.where(a > 0, a, 1e-300))
    pred = np.empty_like(ll)
    filt = np.empty_like(ll)
    scale = np.empty(ll.shape[0])
    for t in range(ll.shape[0]):
        if t == 0 and first_is_start:
            lp = np.log(np.where(s > 0, s, 1e-300))
        else:
            before = np.asarray(prior if t == 0 else filt[t - 1], dtype=float)
            lp = logsumexp(np.log(np.where(before > 0, before, 1e-300))[:, None] + log_a, axis=0)
        pred[t] = np.exp(lp - logsumexp(lp))
        joint = lp + ll[t]
        scale[t] = logsumexp(joint)
        filt[t] = np.exp(joint - scale[t])
    return ForwardResult(pred, filt, scale)


# ---------------------------------------------------------------------------------------- EM + pipeline
def fit_em(obs: NDArray[np.float64], n_states: int, seed: int, max_iter: int, tol: float,
           feature_order: tuple[str, ...]) -> HMMParameters:
    """Fit diagonal-Gaussian EM with hmmlearn and export it to the package-independent schema."""
    import hmmlearn
    from hmmlearn.hmm import GaussianHMM

    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=max_iter, tol=tol, random_state=seed,
                        min_covar=VARIANCE_FLOOR)
    model.fit(np.asarray(obs, dtype=float))
    cov = np.asarray(model.covars_)
    variances = np.array([np.diag(c) for c in cov]) if cov.ndim == 3 else cov
    return HMMParameters(tuple(model.startprob_), tuple(map(tuple, model.transmat_)), tuple(map(tuple, model.means_)),
                         tuple(map(tuple, variances)), feature_order, bool(model.monitor_.converged),
                         f"hmmlearn=={hmmlearn.__version__}")


@dataclass(frozen=True)
class HMMPipelineResult:
    raw: HMMParameters
    aligned: HMMParameters
    permutation: tuple[int, ...]
    start_fallback_stationary: bool
    train: ForwardResult
    boundary_posterior: NDArray[np.float64]
    oos: ForwardResult | None
    first_oos_predicted: NDArray[np.float64] | None


def stationary_distribution(a: NDArray[np.float64]) -> NDArray[np.float64]:
    w, v = np.linalg.eig(a.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    return pi / pi.sum()


def run_hmm_pipeline(train_obs: NDArray[np.float64], oos_obs: NDArray[np.float64] | None, *, n_states: int, seed: int,
                     max_iter: int, tol: float, feature_order: tuple[str, ...],
                     raw_override: HMMParameters | None = None) -> HMMPipelineResult:
    """The indivisible post-refit pipeline of spec 7.5 steps 1-8. ``raw_override`` lets tests inject parameters."""
    raw = raw_override or fit_em(train_obs, n_states, seed, max_iter, tol, feature_order)         # 1-2 fit, export
    validate_parameters(raw, np.asarray(train_obs).shape[1])                                       # 3 validate
    aligned, perm = align_states(raw)                                                              # 4 align
    s = np.array(aligned.start)
    fallback = not (np.all(np.isfinite(s)) and np.all(s >= 0) and abs(s.sum() - 1.0) < 1e-9)
    if fallback:  # a stationary vector is allowed only as a reported fallback for an invalid fitted start (spec 7.5)
        aligned = HMMParameters(tuple(stationary_distribution(np.array(aligned.transition))), aligned.transition,
                                aligned.means, aligned.variances, aligned.feature_order, aligned.converged, aligned.fitter)
    train = forward_filter(aligned, train_obs, first_is_start=True)                               # 5-6 init + filter window
    boundary = train.filtered[-1].copy()                                                           # 7 boundary posterior
    oos = first = None
    if oos_obs is not None and len(oos_obs):                                                       # 8 first OOS from boundary
        oos = forward_filter(aligned, oos_obs, prior=boundary, first_is_start=False)
        first = oos.predicted[0]
    return HMMPipelineResult(raw, aligned, perm, fallback, train, boundary, oos, first)


def p_high(posterior: NDArray[np.float64]) -> float:
    """Probability of the highest-volatility (last aligned) state, i.e. ``high_risk``."""
    return float(np.asarray(posterior)[-1])


# ---------------------------------------------------------------------------------------- fixed fixture regime table
@dataclass(frozen=True)
class FixedRegimeTable:
    """Spec 7.5 schema: ``available_at_utc,state_0_probability,state_1_probability,expected_label``."""

    available: np.ndarray  # datetime64[ns] UTC
    probabilities: np.ndarray  # (rows, 2) aligned state order: state_0 = low_risk
    labels: tuple[str, ...]

    COLUMNS = ("available_at_utc", "state_0_probability", "state_1_probability", "expected_label")

    @classmethod
    def load(cls, path: Path) -> FixedRegimeTable:
        df = pd.read_csv(path)
        if tuple(df.columns) != cls.COLUMNS:
            raise ValueError(f"fixed regime table columns must be {cls.COLUMNS}")
        av = pd.to_datetime(df.available_at_utc, utc=True)
        if not av.is_monotonic_increasing or av.duplicated().any():
            raise ValueError("fixed regime table must have strictly increasing available_at_utc")
        pr = df[["state_0_probability", "state_1_probability"]].to_numpy(float)
        if not np.all(np.isfinite(pr)) or np.any(pr < 0) or not np.allclose(pr.sum(axis=1), 1.0, atol=1e-9):
            raise ValueError("fixed regime probabilities must be finite, nonnegative and sum to one")
        if not set(df.expected_label) <= {"low_risk", "high_risk"}:
            raise ValueError("unknown expected_label")
        return cls(av.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]"), pr, tuple(df.expected_label))

    def asof(self, cutoff: pd.Timestamp) -> tuple[np.ndarray, str]:
        """Latest row with ``available_at_utc <= cutoff``; a cutoff before the first row has no regime (error)."""
        i = int(np.searchsorted(self.available, np.datetime64(cutoff.tz_convert("UTC").tz_localize(None)), side="right")) - 1
        if i < 0:
            raise LookupError(f"no regime row available by {cutoff}")
        return self.probabilities[i], self.labels[i]
