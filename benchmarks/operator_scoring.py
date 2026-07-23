# benchmarks/operator_scoring.py

from collections.abc import Mapping

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Four metric primitives for the two extended composite score axes:
#
#   long-horizon axes  : agg_horizon_instab, agg_horizon_sens
#   SSL-criterion axes : sc, vamp2
#
# How they feed the scorer:
#   * sc / vamp2 are per-data-draw scalars -> append them to each trial record,
#     exactly like spurious_ref_count.  kernel_spectral_score aggregates the
#     per-trial `sc` / `vamp2` columns automatically (mean over trials).
#   * horizon terms need the trial *dispersion* of each eigenvalue, so they are
#     candidate-level by nature (one value per kernel x kind x method, not per
#     trial).  Compute them once per run from the same per-mode records that
#     feed analyse_spectrum, then hand the small table to kernel_spectral_score
#     via `extra_metrics=` (still one call, still within the experiment).
# ---------------------------------------------------------------------------


# ---- defaults -------------------------------------------------------------
MODE_W = {1: 0.5, 2: 0.3, 3: 0.2}  # eigenfunction weights, horizon notebook
H = 50  # forecast horizon in lag steps
R_VAMP = 3  # VAMP-2 rank (modes 1..3)
N_VAMP = 400  # subsample for the whitened SVD (n^3 cost)
EPS_WHITEN = 1e-6  # ridge on the whitening eigenvalues


# ===========================================================================
# --- long-horizon stability / sensitivity ----------------------------------
# ===========================================================================
def horizon_mode_terms(lam, sigma, H=H):
    r"""Reference-free long-horizon terms for ONE eigenmode.

    Given the trial-mean estimated eigenvalue :math:`\hat\lambda` and its
    trial dispersion :math:`\sigma`, over a forecast horizon of ``H`` lag steps:

    .. math::

        \text{instab} = \max(0,\ |\hat\lambda| - 1), \qquad
        \text{sens}   = \frac{1}{H}\sum_{t=1}^{H} t\,|\hat\lambda|^{\,t-1}\,\sigma .

    ``instab`` penalises spectra that blow up (magnitude > 1); ``sens`` is the
    mean derivative of the ``t``-step propagator w.r.t. eigenvalue error, i.e.
    how fast trial noise is amplified along the horizon.
    """
    t = np.arange(1, H + 1)
    instab = max(0.0, abs(lam) - 1.0)
    sens = float(np.mean(t * np.abs(lam) ** (t - 1) * sigma))
    return instab, sens


def horizon_terms(
    modes_df, H=H, mode_weights=MODE_W, group_cols=("kernel", "kind", "method")
):
    r"""Aggregate long-horizon terms per candidate from per-mode trial records.

    ``modes_df`` is the per-mode metrics table (the same records handed to
    ``analyse_spectrum``); it must contain ``eigenfunction_id``,
    ``est_eig_real``, ``est_eig_imag`` and the ``group_cols``.  For each mode
    the trial-mean eigenvalue and trial dispersion are formed, converted to
    per-mode terms via :func:`horizon_mode_terms`, then combined as a
    weighted average over modes.

    Returns a DataFrame with ``group_cols`` + ``agg_horizon_instab`` +
    ``agg_horizon_sens`` (one row per candidate) -- pass straight to
    ``kernel_spectral_score(..., extra_metrics=...)``.
    """
    group_cols = list(group_cols)
    df = modes_df[modes_df["eigenfunction_id"].isin(mode_weights)].copy()
    df["est"] = df["est_eig_real"] + 1j * df["est_eig_imag"]

    rows = []
    for keys, g in df.groupby(group_cols):
        instab = sens = wsum = 0.0
        for mode, gm in g.groupby("eigenfunction_id"):
            w = mode_weights[mode]
            lam = gm["est"].mean()  # trial-mean eigenvalue
            sigma = (
                np.sqrt(
                    gm["est_eig_real"].std() ** 2  # trial dispersion
                    + gm["est_eig_imag"].std() ** 2
                )
                if len(gm) > 1
                else 0.0
            )
            di, ds = horizon_mode_terms(lam, sigma, H=H)
            instab += w * di
            sens += w * ds
            wsum += w
        if wsum > 0:
            keys = keys if isinstance(keys, tuple) else (keys,)
            rows.append(
                dict(
                    zip(group_cols, keys),
                    agg_horizon_instab=instab / wsum,
                    agg_horizon_sens=sens / wsum,
                )
            )
    return pd.DataFrame(rows)


# ===========================================================================
# --- SSL selection criteria: spectral-contrastive loss & VAMP-2 ------------
# ---------------------------------------------------------------------------
# Two input regimes:
#
#   * ENCODER features  phi_X, phi_Y  (shape N x D)  -> use kooplearn's NATIVE
#     losses `SpectralContrastiveLoss` / `VampLoss`, via `sc_score` / `vamp2_score`
#     below.  This is the reference implementation used to *train* the encoders.
#
#   * KERNEL Gram matrices  K = K(X, Y)  (shape N x N)  -> no explicit finite
#     features, so the native (feature-space) losses do not apply directly.  The
#     kernel-space `sc_criterion` / `vamp2_r` below are the Gram-matrix analogues:
#       - `sc_criterion` is *provably identical* to native SpectralContrastiveLoss
#         with K = phi(X) phi(Y)^T  (verified to machine precision).
#       - `vamp2_r` is a bounded, rank-r, canonical-correlation VAMP-2 in kernel
#         space; it is the natural RKHS counterpart of native VampLoss(p=2), which
#         instead sums *all* squared singular values of the feature-covariance
#         whitened operator.
# ===========================================================================

# ---- native kooplearn losses (soft torch/jax dependency) ------------------
# The native losses live behind torch/jax extras.  Import them if available and
# otherwise fall back to numpy reimplementations that reproduce kooplearn's
# formulas exactly (`_sc_features` verified identical; `_vamp_features` mirrors
# kooplearn.torch.nn._functional.vamp_loss).
_KL_BACKEND = None
try:  # preferred: torch backend
    from kooplearn.torch.nn import (
        SpectralContrastiveLoss as _SCLoss,
        VampLoss as _VampLoss,
    )
    import torch as _torch

    _KL_BACKEND = "torch"
except Exception:  # pragma: no cover - optional deps
    try:  # jax backend
        from kooplearn.jax.nn import (
            SpectralContrastiveLoss as _SCLoss,
            VampLoss as _VampLoss,
        )
        import jax.numpy as _jnp

        _KL_BACKEND = "jax"
    except Exception:
        _KL_BACKEND = None


def _sc_features(x, y):
    """numpy reimplementation of kooplearn SpectralContrastiveLoss (lower=better)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    npts, dim = x.shape
    diag = 2.0 * np.mean(x * y) * dim
    sq = (x @ y.T) ** 2
    off = np.mean(np.triu(sq, 1) + np.tril(sq, -1)) * npts / (npts - 1)
    return float(off - diag)


def _covariance(X, Y=None, center=True):
    """kooplearn-style (cross-)covariance: (X/sqrt(N))^T (Y/sqrt(N)), optionally centered."""
    n = np.sqrt(X.shape[0])
    X = np.asarray(X, float) / n
    if Y is None:
        if center:
            X = X - X.mean(0, keepdims=True)
        return X.T @ X
    Y = np.asarray(Y, float) / n
    if center:
        X = X - X.mean(0, keepdims=True)
        Y = Y - Y.mean(0, keepdims=True)
    return X.T @ Y


def _vamp_features(x, y, schatten_norm=2, center_covariances=True):
    """numpy reimplementation of kooplearn VampLoss VAMP-p score (higher=better).

    Returns +score (kooplearn's `vamp_loss` returns the negative, as a loss)."""
    cx = _covariance(x, center=center_covariances)
    cy = _covariance(y, center=center_covariances)
    cxy = _covariance(x, y, center=center_covariances)
    if schatten_norm == 2:
        m_x = np.linalg.lstsq(cx, cxy, rcond=None)[0]
        m_y = np.linalg.lstsq(cy, cxy.T, rcond=None)[0]
        return float(np.trace(m_x @ m_y))
    elif schatten_norm == 1:

        def _inv_sqrt(C):
            w, V = np.linalg.eigh(C)
            w = np.clip(w, np.finfo(float).eps * C.shape[0], None)
            return (V / np.sqrt(w)) @ V.T

        A = _inv_sqrt(cx) @ cxy @ _inv_sqrt(cy)
        return float(np.sum(np.linalg.svd(A, compute_uv=False)))
    raise NotImplementedError(f"Schatten norm {schatten_norm} not implemented")


def sc_score(phi_X, phi_Y):
    r"""Spectral-contrastive score on ENCODER features (lower = better).

    Thin wrapper over kooplearn's native ``SpectralContrastiveLoss`` when the
    torch/jax extra is installed; otherwise an exact numpy reimplementation.

    ``phi_X``, ``phi_Y`` are the encoded lagged pairs, shape ``(N, D)``.
    """
    if _KL_BACKEND == "torch":
        return float(
            _SCLoss()(
                _torch.as_tensor(np.asarray(phi_X), dtype=_torch.float64),
                _torch.as_tensor(np.asarray(phi_Y), dtype=_torch.float64),
            )
        )
    if _KL_BACKEND == "jax":
        return float(_SCLoss()(_jnp.asarray(phi_X), _jnp.asarray(phi_Y)))
    return _sc_features(phi_X, phi_Y)


def vamp2_score(phi_X, phi_Y, *, schatten_norm=2, center_covariances=True):
    r"""VAMP-p score on ENCODER features (higher = better).

    Thin wrapper over kooplearn's native ``VampLoss`` (returns the *negated*
    loss, i.e. the score) when torch/jax is installed; otherwise an exact numpy
    reimplementation.  ``schatten_norm=2`` gives the standard VAMP-2 score.
    """
    if _KL_BACKEND == "torch":
        loss = _VampLoss(
            schatten_norm=schatten_norm, center_covariances=center_covariances
        )(
            _torch.as_tensor(np.asarray(phi_X), dtype=_torch.float64),
            _torch.as_tensor(np.asarray(phi_Y), dtype=_torch.float64),
        )
        return -float(loss)
    if _KL_BACKEND == "jax":
        loss = _VampLoss(
            schatten_norm=schatten_norm, center_covariances=center_covariances
        )(_jnp.asarray(phi_X), _jnp.asarray(phi_Y))
        return -float(loss)
    return _vamp_features(phi_X, phi_Y, schatten_norm, center_covariances)


def sc_criterion(K_xy_val):
    r"""Spectral-contrastive loss from the cross-kernel Gram matrix (lower = better).

    .. math::

        \mathrm{SC} = \frac{1}{N(N-1)}\sum_{i\neq j} K_{ij}^2
                      - \frac{2}{N}\sum_i K_{ii},

    with :math:`K = K(X_\text{val}, Y_\text{val})` the cross-kernel Gram matrix.

    This is exactly kooplearn's native ``SpectralContrastiveLoss`` evaluated with
    :math:`K = \phi(X)\,\phi(Y)^\top` (verified identical to machine precision) --
    it is the kernel-space form used when explicit features are unavailable.  For
    encoder features prefer :func:`sc_score`, which calls the native loss.
    """
    K = np.asarray(K_xy_val, float)
    N = K.shape[0]
    diag = np.diag(K)
    off_sq = (K**2).sum() - (diag**2).sum()
    return float(off_sq / (N * (N - 1)) - 2.0 * diag.mean())


def vamp2_r(K_X, K_Y, K_XY, r=R_VAMP, eps=EPS_WHITEN):
    r"""VAMP-2(r): sum of the top-``r`` squared singular values of the whitened
    cross-operator, estimated from Gram matrices (higher = better).

    .. math::

        \mathrm{VAMP2}_r = \sum_{k=1}^{r} \sigma_k^2\!\big(
            K_X^{-1/2}\, K_{XY}\, K_Y^{-1/2}\big),

    the singular values being the estimated canonical correlations
    (clipped to :math:`[0, 1]`).  ``eps`` ridges the whitening eigenvalues.

    Relation to native ``VampLoss(schatten_norm=2)``: the native score sums *all*
    squared singular values of the **feature-covariance** whitened operator
    :math:`(x^\top x)^{\dagger/2} x^\top y (y^\top y)^{\dagger/2}`, uncentered/uncipped.
    This kernel-space variant is deliberately rank-``r`` (matching modes 1..r) and
    clipped to canonical correlations, so it is bounded in :math:`[0, r]`.  For
    encoder features prefer :func:`vamp2_score`, which calls the native loss.
    """
    n = K_X.shape[0]

    def inv_sqrt(K):
        w, V = np.linalg.eigh(K / n)
        w = np.clip(w, eps, None)
        return (V / np.sqrt(w)) @ V.T / np.sqrt(n)

    A = inv_sqrt(K_X) @ K_XY @ inv_sqrt(K_Y)
    s = np.linalg.svd(A, compute_uv=False)
    s = np.clip(s, 0, 1.0 + 1e-6)  # canonical correlations
    return float(np.sum(s[:r] ** 2))


def kernel_selection_criteria(
    K_fn, X_tr, Y_tr, X_val, Y_val, *, r=R_VAMP, n_vamp=N_VAMP, eps=EPS_WHITEN
):
    r"""Both SSL criteria for ONE **kernel** on ONE data draw (Gram-matrix regime).

    ``K_fn(A, B)`` must return the Gram matrix between row-sets ``A`` and ``B``.
    VAMP-2 uses a size-``n_vamp`` subsample of the training pairs (the whitening
    SVD is :math:`O(n^3)`); SC uses the full validation pairs.

    For learned **encoders** (explicit features), use
    :func:`feature_selection_criteria`, which calls kooplearn's native losses.

    Returns ``{"sc": ..., "vamp2": ...}`` -- merge into the trial record so the
    scorer picks them up as per-trial ``sc`` / ``vamp2`` columns.
    """
    sub = slice(0, n_vamp)
    K_x = K_fn(X_tr[sub], X_tr[sub])
    K_y = K_fn(Y_tr[sub], Y_tr[sub])
    K_xy = K_fn(X_tr[sub], Y_tr[sub])
    return {
        "sc": sc_criterion(K_fn(X_val, Y_val)),
        "vamp2": vamp2_r(K_x, K_y, K_xy, r=r, eps=eps),
    }


def feature_selection_criteria(
    phi_X_tr,
    phi_Y_tr,
    phi_X_val,
    phi_Y_val,
    *,
    schatten_norm=2,
    center_covariances=True,
):
    r"""Both SSL criteria for ONE **encoder** on ONE data draw (feature regime).

    ``phi_*`` are encoded lagged pairs, shape ``(N, D)``.  Uses kooplearn's native
    ``SpectralContrastiveLoss`` / ``VampLoss`` (or the exact numpy fallback).  SC is
    evaluated on the validation pairs; VAMP-2 on the training pairs.

    Returns ``{"sc": ..., "vamp2": ...}`` for merging into the trial record.
    """
    return {
        "sc": sc_score(phi_X_val, phi_Y_val),
        "vamp2": vamp2_score(
            phi_X_tr,
            phi_Y_tr,
            schatten_norm=schatten_norm,
            center_covariances=center_covariances,
        ),
    }


# ---------------------------------------------------------------------------
# What it includes:
#   * per-metric log10 transform for heavy-tailed axes (LOG_METRICS)
#   * within-pool z-score normalisation (groupby -> transform), not global
#   * sign flip for larger-is-better axes (mean_spectral_gap, vamp2)
#   * v3 weights (Kostic + long-horizon + SSL-criterion axes)
#   * missing / degenerate axes are NEUTRAL (z = 0), so heterogeneous candidate
#     sets pool fairly (e.g. kernels without VAMP alongside those with it)
#   * hard-constraint admissibility + within-pool ranking (inadmissible last)
#
# Lower composite score = better:
# `rank` is the within-pool selection rank
# (1 = best in its pool); `rank_overall` is the global order for convenience.
# ---------------------------------------------------------------------------


# ===========================================================================
# ------------------------ THE scoring configuration ------------------------
# ===========================================================================
GRAND_WEIGHTS = {
    # Kostic spectral axes (frozen originals)
    "agg_bias_mean": 1.0,
    "agg_dist_mean": 1.0,
    "agg_spurious_mean": 1.0,
    "agg_trunc_mean": 0.5,
    "agg_bias_std": 0.25,
    "agg_spurious_std": 0.25,
    "mean_spurious_ref_count": 0.5,
    "mean_spurious_residual_count": 1.0,
    "mean_spectral_gap": 0.5,
    # long-horizon axes
    "agg_horizon_instab": 1.0,
    "agg_horizon_sens": 1.0,
    # SSL-criterion axes
    "sc": 1.0,
    "vamp2": 0.25,
}
# v2 = grand score WITHOUT the SSL-criterion axes
V2_WEIGHTS = {k: v for k, v in GRAND_WEIGHTS.items() if k not in ("sc", "vamp2")}

LARGER_IS_BETTER = {"mean_spectral_gap", "vamp2"}
LOG_METRICS = {
    "agg_bias_mean",
    "agg_dist_mean",
    "agg_bias_std",
    "agg_spurious_std",
    "agg_trunc_mean",
    "agg_horizon_sens",
}
HARD_CONSTRAINTS = {"max_spurious_ref_count": 4, "max_spurious_residual_count": 5}

# sentinel to distinguish "caller did not pass hard_constraints" (-> use
# grand defaults) from "caller explicitly passed None / {}" (-> disable).
_DEFAULT = object()


# ===========================================================================
# --- normalisation (the improved method: log -> grouped z-score) -----------
# ===========================================================================
def _normalise_series(s, method="zscore", larger_is_better=False, log=False):
    """Normalise ONE series (a metric column, or a single pool's slice of it).

    Pipeline mirrors grand_kernel_score._norm:
        to_numeric -> drop +/-inf -> optional log10(clip 1e-16)
                   -> optional sign flip (larger-is-better) -> z/minmax/rank.

    Degenerate groups (all-NaN, or zero / non-finite spread) collapse to 0.0,
    i.e. a *neutral* contribution.  All three methods are oriented so that a
    larger normalised value means "worse" (consistent with lower-score-better).
    """
    x = pd.to_numeric(s, errors="coerce").astype(float)
    x = x.replace([np.inf, -np.inf], np.nan)

    if log:
        x = np.log10(x.clip(lower=1e-16))
    if larger_is_better:
        x = -x

    m = x.notna()
    out = pd.Series(0.0, index=x.index, dtype=float)
    if m.sum() == 0:
        return out

    vals = x[m].to_numpy(dtype=float)

    if method == "zscore":
        mu = np.nanmean(vals)
        sd = np.nanstd(vals)  # population std (ddof=0), as in the notebook
        if np.isfinite(sd) and sd > 0:
            out.loc[m] = (vals - mu) / sd
        # else: leave neutral zeros

    elif method == "minmax":
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        if np.isfinite(hi - lo) and hi > lo:
            out.loc[m] = (vals - lo) / (hi - lo)

    elif method == "rank":
        r = pd.Series(vals).rank(method="average").to_numpy(dtype=float)
        denom = r.max() - 1.0
        if denom > 0:
            out.loc[m] = (r - 1.0) / denom  # ascending: larger value -> worse

    else:
        raise ValueError(f"Unknown normalise method: {method}")

    return out


def _normalise_grouped(df, metric, pool_cols, *, method, larger_is_better, log):
    """Apply `_normalise_series` within each pool (or globally if no pool)."""
    kwargs = dict(method=method, larger_is_better=larger_is_better, log=log)
    if pool_cols:
        return df.groupby(list(pool_cols), dropna=False)[metric].transform(
            lambda col: _normalise_series(col, **kwargs)
        )
    return _normalise_series(df[metric], **kwargs)


# ===========================================================================
# ----------------------- composite scorer ----------------------------------
# ===========================================================================
def composite_score(
    summary,
    trials_df=None,
    *,
    group_cols=("kernel", "kind", "method"),
    candidate_col="kernel",
    pool_cols=None,
    selected_modes=None,
    mode_weights=None,
    normalise="zscore",
    metric_weights=None,
    log_metrics=LOG_METRICS,
    larger_is_better=None,
    hard_constraints=_DEFAULT,
    extra_metrics=None,
    use_trial_metrics=True,
    score_name="grand_score",
):
    """Score kernels across 3 axes: spectral metrics, long horizon, and SSL criteria.

    Parameters
    ----------
    summary : DataFrame
        Per-mode table from `analyse_spectrum` (one row per
        kernel x kind x method x eigenfunction_id) with columns
        bias_mean/bias_std/dist_mean/trunc_mean/spurious_mean/spurious_std.
    trials_df : DataFrame, optional
        Per-trial table.  Any of these columns, if present, are aggregated to
        per-candidate axes: spurious_ref_count, spurious_residual_count,
        spectral_gap, rank, horizon_instab, horizon_sens, sc, vamp2.
    group_cols : tuple
        Candidate granularity (one scored row per unique combination).
    candidate_col : str
        The axis that *varies within a pool* (the thing being selected between).
    pool_cols : tuple, optional
        Keys defining the normalisation / ranking pool.  Defaults to
        `group_cols` minus `candidate_col` (i.e. compare kernels within each
        kind x method cell -- the grand per-family semantics).  Pass e.g.
        ("system", "kind") to reproduce grand's cross-family "pooled" mode.
    metric_weights : dict, optional
        Defaults to `GRAND_WEIGHTS` (v3).  Use `V2_WEIGHTS` to drop the SSL axes.
    extra_metrics : DataFrame, optional
        Pre-computed per-candidate axes (e.g. agg_horizon_*, sc, vamp2) merged
        on the intersection of `group_cols`.  Anything still missing stays
        neutral, exactly as in the notebook.

    Returns
    -------
    (mode_agg_df, kernel_scores_df)
        `kernel_scores_df` carries `composite_score`, `admissible`,
        `constraint_violations`, `rank` (within pool) and `rank_overall`.
    """
    summary = summary.copy()
    group_cols = list(group_cols)

    if pool_cols is None:
        pool_cols = [c for c in group_cols if c != candidate_col]
    pool_cols = list(pool_cols)

    if larger_is_better is None:
        larger_is_better = set(LARGER_IS_BETTER)
    larger_is_better = set(larger_is_better)
    log_metrics = set(log_metrics or ())

    if metric_weights is None:
        metric_weights = dict(GRAND_WEIGHTS)

    if hard_constraints is _DEFAULT:
        hard_constraints = dict(HARD_CONSTRAINTS)

    # ---- mode selection + mode weights ------------------------------------
    if selected_modes is not None:
        summary = summary[summary["eigenfunction_id"].isin(selected_modes)].copy()
    if summary.empty:
        raise ValueError("No rows remain in summary after filtering selected_modes.")

    if mode_weights is None:
        mode_weights = {m: 1.0 for m in sorted(summary["eigenfunction_id"].unique())}
    elif isinstance(mode_weights, Mapping):
        mode_weights = dict(mode_weights)
    else:
        mode_weights = {m: w for m, w in mode_weights}

    summary["mode_weight"] = summary["eigenfunction_id"].map(mode_weights).fillna(0.0)
    if (summary["mode_weight"] < 0).any():
        raise ValueError("mode_weights must be nonnegative.")

    # ---- stage 1: aggregate per-mode metrics -> per-candidate axes ---------
    def _wavg(g, col):
        w = g["mode_weight"].to_numpy(dtype=float)
        x = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & (w > 0)
        if not ok.any():
            return np.nan
        return float(np.average(x[ok], weights=w[ok]))

    mode_agg_df = (
        summary.groupby(group_cols, as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "n_modes_used": int(g["eigenfunction_id"].nunique()),
                    "weight_sum": float(g["mode_weight"].sum()),
                    "agg_bias_mean": _wavg(g, "bias_mean"),
                    "agg_bias_std": _wavg(g, "bias_std"),
                    "agg_dist_mean": _wavg(g, "dist_mean"),
                    "agg_trunc_mean": _wavg(g, "trunc_mean"),
                    "agg_spurious_mean": _wavg(g, "spurious_mean"),
                    "agg_spurious_std": _wavg(g, "spurious_std"),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    trials_df = trials_df.copy() if trials_df is not None else None

    # ---- stage 2: aggregate the fixed per-trial axes (only those present) --
    if use_trial_metrics and trials_df is not None:
        tgroup = [c for c in group_cols if c in trials_df.columns]
        # map trial column -> (output name, aggfunc); include only if present
        trial_spec = {
            "spurious_ref_count": [("mean_spurious_ref_count", "mean")],
            "spurious_residual_count": [("mean_spurious_residual_count", "mean")],
            "spectral_gap": [
                ("mean_spectral_gap", "mean"),
                ("std_spectral_gap", "std"),
            ],
            "rank": [("mean_rank", "mean")],
        }
        named_agg = {
            out: (src, fn)
            for src, outs in trial_spec.items()
            if src in trials_df.columns
            for out, fn in outs
        }
        if tgroup and named_agg:
            trial_agg = trials_df.groupby(tgroup, as_index=False).agg(**named_agg)
            mode_agg_df = mode_agg_df.merge(trial_agg, on=tgroup, how="left")

    # ---------------- stage 2b: optional horizon / SSL axes -----------------
    # canonical output name -> accepted source aliases (raw or pre-aggregated).
    # Resolution precedence per axis:
    #   1. already on mode_agg_df (produced upstream)   -> keep
    #   2. per-trial column in trials_df                -> mean over trials
    #   3. per-mode column in summary                   -> mode-weighted average
    #   4. extra_metrics (explicit override)            -> merged last, below
    # Anything still unresolved stays absent -> neutral (z = 0) in the score.
    OPTIONAL_AXES = {
        "agg_horizon_instab": ("agg_horizon_instab", "horizon_instab"),
        "agg_horizon_sens": ("agg_horizon_sens", "horizon_sens"),
        "sc": ("sc",),
        "vamp2": ("vamp2",),
    }
    for out_name, aliases in OPTIONAL_AXES.items():
        if out_name in mode_agg_df.columns and mode_agg_df[out_name].notna().any():
            continue  # (1) already there

        # (2) per-trial source
        if use_trial_metrics and trials_df is not None:
            tgroup = [c for c in group_cols if c in trials_df.columns]
            tsrc = next((a for a in aliases if a in trials_df.columns), None)
            if tgroup and tsrc is not None:
                agg = (
                    trials_df.groupby(tgroup, as_index=False)[tsrc]
                    .mean()
                    .rename(columns={tsrc: out_name})
                )
                mode_agg_df = mode_agg_df.merge(agg, on=tgroup, how="left")
                continue

        # (3) per-mode source in summary (mode-weighted, like the other axes)
        ssrc = next((a for a in aliases if a in summary.columns), None)
        if ssrc is not None:
            wa = (
                summary.groupby(group_cols, as_index=False)
                .apply(
                    lambda g, c=ssrc: pd.Series({out_name: _wavg(g, c)}),
                    include_groups=False,
                )
                .reset_index(drop=True)
            )
            mode_agg_df = mode_agg_df.merge(wa, on=group_cols, how="left")

    # ---- optional: explicit pre-computed per-candidate axes (override) -----
    if extra_metrics is not None:
        on = [
            c
            for c in group_cols
            if c in extra_metrics.columns and c in mode_agg_df.columns
        ]
        if on:
            add = [c for c in extra_metrics.columns if c not in on]
            # drop any placeholder columns we are about to override
            mode_agg_df = mode_agg_df.drop(
                columns=[c for c in add if c in mode_agg_df.columns]
            )
            mode_agg_df = mode_agg_df.merge(extra_metrics, on=on, how="left")

    # ---- stage 3: normalise (within pool) + weighted composite ------------
    df = mode_agg_df.copy().reset_index(drop=True)

    used_metrics = []
    for metric, weight in metric_weights.items():
        if weight == 0 or metric not in df.columns:
            continue
        if not pd.to_numeric(df[metric], errors="coerce").notna().any():
            continue  # axis entirely absent -> neutral (skip)
        df[f"{metric}_norm"] = _normalise_grouped(
            df,
            metric,
            pool_cols,
            method=normalise,
            larger_is_better=(metric in larger_is_better),
            log=(metric in log_metrics),
        )
        used_metrics.append(metric)

    score = np.zeros(len(df), dtype=float)
    for metric in used_metrics:
        score += metric_weights[metric] * df[f"{metric}_norm"].to_numpy(dtype=float)
    df[score_name] = score
    df["composite_score"] = score  # back-compatible alias
    df["used_metrics"] = ", ".join(used_metrics)

    # ---- stage 4: hard-constraint admissibility ---------------------------
    admissible = np.ones(len(df), dtype=bool)
    viol = [[] for _ in range(len(df))]

    def _mark(mask, label):
        nonlocal admissible
        if mask is None:
            return
        mask = mask.fillna(False).to_numpy()
        admissible &= ~mask
        for i in np.where(mask)[0]:
            viol[i].append(label)

    hc = hard_constraints or {}
    _upper = {  # constraint key -> (column, label)
        "max_spurious_ref_count": ("mean_spurious_ref_count", "spurious_ref"),
        "max_spurious_residual_count": (
            "mean_spurious_residual_count",
            "spurious_residual",
        ),
        "max_dist_mean": ("agg_dist_mean", "distortion"),
        "max_bias_mean": ("agg_bias_mean", "bias"),
        "max_trunc_mean": ("agg_trunc_mean", "truncation"),
    }
    for key, (col, label) in _upper.items():
        if key in hc and col in df.columns:
            _mark(pd.to_numeric(df[col], errors="coerce") > hc[key], label)
    if "min_spectral_gap" in hc and "mean_spectral_gap" in df.columns:
        _mark(
            pd.to_numeric(df["mean_spectral_gap"], errors="coerce")
            < hc["min_spectral_gap"],
            "gap",
        )

    df["admissible"] = admissible
    df["constraint_violations"] = [",".join(v) for v in viol]

    # ---- stage 5: ranking (within pool; inadmissible sorted last) ---------
    df["_infeasible"] = (~df["admissible"]).astype(int)
    sort_keys = pool_cols + ["_infeasible", "composite_score"]
    srt = df.sort_values(sort_keys, kind="mergesort")

    if pool_cols:
        df["rank"] = (srt.groupby(pool_cols, dropna=False).cumcount() + 1).reindex(
            df.index
        )
    else:
        df["rank"] = pd.Series(np.arange(1, len(df) + 1), index=srt.index).reindex(
            df.index
        )

    srt_g = df.sort_values(["_infeasible", "composite_score"], kind="mergesort")
    df["rank_overall"] = pd.Series(
        np.arange(1, len(df) + 1), index=srt_g.index
    ).reindex(df.index)
    df = df.drop(columns="_infeasible")

    kernel_scores_df = (
        df.sort_values(pool_cols + ["rank"], kind="mergesort").reset_index(drop=True)
        if pool_cols
        else df.sort_values("rank_overall", kind="mergesort").reset_index(drop=True)
    )
    return mode_agg_df, kernel_scores_df


# ===========================================================================
# ----------------- sensitivity study around baseline weights ---------------
# ===========================================================================
def run_weight_sensitivity(
    summary,
    trials_df,
    *,
    base_weights=None,
    vary_metrics=("agg_spurious_mean", "mean_spurious_residual_count"),
    scales=(0.8, 1.0, 1.2),
    selected_modes=(1, 2, 3),
    mode_weights=None,
    normalise="zscore",
    group_cols=("kernel", "kind", "method"),
    pool_cols=None,
    hard_constraints=_DEFAULT,
    extra_metrics=None,
    top_k=5,
):
    if base_weights is None:
        base_weights = dict(GRAND_WEIGHTS)

    rows, top_rows = [], []
    for metric in vary_metrics:
        for scale in scales:
            w = dict(base_weights)
            w[metric] = base_weights.get(metric, 0.0) * scale
            _, scores = composite_score(
                summary,
                trials_df=trials_df,
                group_cols=group_cols,
                pool_cols=pool_cols,
                selected_modes=list(selected_modes),
                mode_weights=mode_weights,
                normalise=normalise,
                metric_weights=w,
                hard_constraints=hard_constraints,
                extra_metrics=extra_metrics,
            )
            # global order so "best" is unambiguous regardless of pooling
            scores_sorted = scores.sort_values(
                ["admissible", "rank_overall"], ascending=[False, True]
            ).reset_index(drop=True)
            top = scores_sorted.head(top_k).copy()
            best_row = top.iloc[0]
            rows.append(
                {
                    "varied_metric": metric,
                    "scale": scale,
                    "best_kernel": best_row.get("kernel"),
                    "best_kind": best_row.get("kind"),
                    "best_method": best_row.get("method"),
                    "best_score": best_row["composite_score"],
                    "best_rank": best_row["rank"],
                    "best_admissible": best_row["admissible"],
                    "top5_signature": " | ".join(
                        f"{r.get('kernel')} / {r.get('method')} (r{int(r['rank_overall'])})"
                        for _, r in top.iterrows()
                    ),
                }
            )
            top["varied_metric"] = metric
            top["scale"] = scale
            top_rows.append(top)

    summary_df = (
        pd.DataFrame(rows)
        .sort_values(["varied_metric", "scale"])
        .reset_index(drop=True)
    )
    top_df = pd.concat(top_rows, ignore_index=True)
    return summary_df, top_df
