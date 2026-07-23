# benchmarks/kernel_spectrum.py
# Implementation of Kostic et al. 2023 metrics for analysing kernel spectrum

import re
from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from operator_scoring import MODE_W, H, horizon_terms

from kooplearn._linalg import eigh_rank_reveal, spd_neg_pow, weighted_norm


# ===========================
# --- operator norm error ---
# ===========================
def operator_norm_error(true_operator: np.ndarray, estimated_operator: np.ndarray):
    r"""Operator norm error proxy for a Koopman estimator.

    Computes the operator norm discrepancy between the true action
    :math:`A_\pi S` and the estimated action :math:`S \widehat{G}`:

    .. math::

        \mathcal{E}(\widehat{G}) := \|A_\pi S - S \widehat{G}\|.

    Since "kooplearn" does not currently expose :math:`A_\pi` or the embedding
    operator :math:`S` explicitly, this function works with their actions on a
    common finite-dimensional representation. In practice, the caller should pass
    matrices or vectors representing the two quantities to be compared.
    """
    true_operator = np.asanyarray(true_operator)
    estimated_operator = np.asanyarray(estimated_operator)

    if true_operator.shape != estimated_operator.shape:
        raise ValueError(
            "true_operator and estimated_operator must have the same "
            f"shape, got {true_operator.shape} and "
            f"{estimated_operator.shape}."
        )

    diff = true_operator - estimated_operator
    if diff.ndim == 1:
        return float(np.linalg.norm(diff))
    return float(np.linalg.norm(diff, ord=2))


# ==========================
# --- metric distortion ---
# ==========================
def metric_distortion(psi, C):
    r"""Empirical metric distortion :math:`\widehat\eta_i = \|\widehat\psi_i\|_{\mathcal H} /
    \sqrt{\langle \widehat C \widehat\psi_i, \widehat\psi_i\rangle}`.

    Parameters
    ----------
    psi : ndarray, shape (n,) or (n, k)
        Eigenfunction(s) evaluated at the *training* points. If 2D, each
        column is treated as a separate eigenfunction (see `weighted_norm`).
    C : ndarray, shape (n, n)
        Empirical (kernel-based) covariance, i.e. ``model.kernel_X / n_samples``.
    """
    psi = np.asarray(psi)
    n = C.shape[0]

    # ||psi||_H via the reproducing property: needs the *inverse* Gram, since
    # C = K_X / n is the Gram-based covariance, not the RKHS metric itself.
    C_inv = spd_neg_pow(C * n, exponent=-1.0)  # i.e. K_X^{-1}
    rkhs_norm = weighted_norm(psi, M=C_inv)

    # <C psi, psi> = (1/n)||psi(X)||_2^2, i.e. weighted_norm with M=None, squared, over n
    empirical_norm = weighted_norm(psi, M=None) / np.sqrt(n)

    with np.errstate(divide="ignore", invalid="ignore"):
        eta = rkhs_norm / empirical_norm
    eta = np.where(empirical_norm > 0, eta, np.nan)
    return eta if psi.ndim == 2 else float(eta)


# =====================
# --- spectral bias ---
# =====================


# --- truncation helpers ---


def _top_sv(C, r):
    """(r+1)-st eigenvalue of a symmetric PSD matrix, via eigh_rank_reveal."""
    raw_vals, raw_vecs = np.linalg.eigh(np.asarray(C))
    _, top_vals, _ = eigh_rank_reveal(raw_vals, raw_vecs, rank=r + 1)
    if len(top_vals) <= r:
        return 0.0
    return float(top_vals[-1])


def pcr_truncation(C, r):
    r""":math:`\rho_{r+1}(\widehat G^{PCR}) = \sigma_{r+1}(\widehat C)`."""
    return _top_sv(C, r)


# kDMD uses the same (r+1)-st eigenvalue of the empirical covariance as PCR
kdmd_truncation = pcr_truncation


def rrr_truncation(C, T, r, cutoff=None):
    r""":math:`\rho_{r+1}(\widehat G^{RRR}) = \sigma_{r+1}(\widehat C^{-1/2}\widehat T)`."""
    C_inv_sqrt = spd_neg_pow(np.asarray(C), exponent=-0.5, cutoff=cutoff)
    A = C_inv_sqrt @ np.asarray(T)
    svals = np.linalg.svd(A, compute_uv=False)
    if r >= len(svals):
        return 0.0
    return float(svals[r])


# --- bias function ---


def spectral_bias(eigenfunction, C, rho):
    r"""Empirical spectral bias :math:`\hat s_i = \widehat\eta_i \, \rho_{r+1}`."""
    eta = metric_distortion(eigenfunction, C)
    s_hat = eta * rho
    return float(s_hat), eta


# ====================================================
# --- spectral gap (|mu_l - mu_j|) ---
# ====================================================
def koopman_gap(eigs):
    """gap_j = min_{l != j} |mu_l - mu_j| for each eigenvalue in `eigs`.

    Call on the full vals_hat inside the experiment loops
    (all n_components), record per mode alongside spectral_bias.
    """
    eigs = np.atleast_1d(np.asarray(eigs, complex))
    n = len(eigs)
    if n < 2:
        return np.full(n, np.nan)
    D = np.abs(eigs[:, None] - eigs[None, :])
    np.fill_diagonal(D, np.inf)
    return D.min(axis=1)


# =========================================
# --- spurious eigenvalues vs reference ---
# =========================================


def spurious_ref(est, ref, delta):
    dist = np.abs(est[:, None] - ref[None, :])
    return int(np.sum(dist.min(axis=1) > delta))


# ===========================================================================
# --- ResDMD residual & pseudospectra (Colbrook & Townsend 2024) ------------
# ===========================================================================
#
# THE residual used throughout the pipeline is the ResDMD residual.  For a
# candidate eigenpair (lambda, g), with Galerkin matrices G = Psi_X^* W Psi_X,
# A = Psi_X^* W Psi_Y and the third matrix L = Psi_Y^* W Psi_Y,
#
#   res(lambda, g)^2 = g^* [ L - lambda A^* - conj(lambda) A + |lambda|^2 G ] g
#                      / ( g^* G g )                                (Colbrook 3.2)
#
# converges to the true operator residual ||K g - lambda g|| / ||g|| and, by
# Thm 3.1, upper-bounds the distance of lambda to the true spectrum -- a *verified*
# pseudospectral certificate.  Equivalently, in function-value form,
#   res = ||psi(Y) - lambda psi(X)|| / ||psi(X)||,
# which is IDENTICAL for eigenpairs (verified to ~2e-15; the third matrix enters
# as ||psi(Y)||^2 = g^* L g).  There is ONE residual function, `resdmd_residual`,
# computing it either way.  Algorithm 1 accepts eigenpairs with res <= eps (a
# pseudospectral tolerance); `resdmd_pseudospectrum` (Algorithm 2) extends the
# same residual to arbitrary z to trace the pseudospectrum (non-normal systems).


def galerkin_matrices(Phi_X, Phi_Y, weights=None):
    r"""The three ResDMD Galerkin matrices from dictionary evaluations.

    Returns ``(G, A, L)`` with ``G = Psi_X^* W Psi_X``, ``A = Psi_X^* W Psi_Y``,
    ``L = Psi_Y^* W Psi_Y``.  ``Phi_X`` / ``Phi_Y`` are the dictionary/features on
    inputs and one-step outputs, shape ``(n, d)``.  ``L`` is the one extra matrix
    ResDMD needs beyond EDMD (costs nothing -- ``Phi_Y`` is already in hand); it is
    used implicitly by the function-value residual as ``||psi(Y)||^2 = g^* L g``.
    """
    Phi_X = np.asarray(Phi_X)
    Phi_Y = np.asarray(Phi_Y)
    n = Phi_X.shape[0]
    w = (
        np.full(n, 1.0 / n)
        if weights is None
        else np.asarray(weights, float) / np.sum(weights)
    )
    WXt = Phi_X.conj().T * w  # (d, n): columns scaled by the quadrature weights
    G = WXt @ Phi_X
    A = WXt @ Phi_Y
    L = (Phi_Y.conj().T * w) @ Phi_Y
    return G, A, L


def resdmd_residual(
    eigenvalues,
    psi_X=None,
    psi_Y=None,
    eps=None,
    relative=True,
    *,
    eigvecs=None,
    G=None,
    A=None,
    L=None,
):
    r"""THE ResDMD residual per eigenpair (Colbrook eq. 3.2) -- the single residual
    used everywhere in the pipeline.  Computed in either of two mathematically
    identical ways (verified to ~2e-15):

    * **function-value form (default)** -- pass ``psi_X`` / ``psi_Y`` = the
      eigenfunctions evaluated at inputs / one-step outputs, shape ``(n, r)``.
      ``res_i = ||psi_i(Y) - lambda_i psi_i(X)|| / ||psi_i(X)||``.  This is what the
      experiment loops have to hand, so it is the efficient default.
    * **Galerkin form** -- pass ``eigvecs`` ``(d, r)`` and ``G, A, L`` from
      :func:`galerkin_matrices` to evaluate the explicit three-matrix quadratic
      form 3.2 (needed when the features, not the eigenfunction values, are held).

    Returns
    -------
    res : ndarray (r,)                       if ``eps`` is None.
    (n_spurious, res) : (int, ndarray)       if ``eps`` is given -- the ResDMD
        Algorithm-1 result, ``n_spurious = #{res > eps}`` (``eps`` = pseudospectral
        tolerance; lower res = better verified).
    """
    vals = np.atleast_1d(np.asarray(eigenvalues))
    if G is not None:  # explicit Galerkin form
        V = np.asarray(eigvecs)
        res = np.empty(V.shape[1], dtype=float)
        for i in range(V.shape[1]):
            lam = vals[i]
            g = V[:, i]
            R = L - lam * A.conj().T - np.conj(lam) * A + np.abs(lam) ** 2 * G
            num = (g.conj() @ R @ g).real
            den = (g.conj() @ G @ g).real
            res[i] = np.sqrt(max(num, 0.0) / den) if den > 0 else np.nan
    else:  # function-value form (identical for eigenpairs)
        n = psi_X.shape[0]
        resid = np.asarray(psi_Y) - np.asarray(psi_X) * vals[None, :]
        resid_norm = weighted_norm(resid) / np.sqrt(n)
        if relative:
            base = weighted_norm(psi_X) / np.sqrt(n)
            res = np.full_like(resid_norm, np.nan, dtype=float)
            ok = np.isfinite(base) & (base > 0)
            res[ok] = resid_norm[ok] / base[ok]
        else:
            res = resid_norm
    if eps is None:
        return res
    with np.errstate(invalid="ignore"):
        n_spurious = int(np.sum(res > eps))  # NaN > eps -> False, so NaNs are ignored
    return n_spurious, res


def resdmd_verified_mask(residual, eps):
    """Colbrook Algorithm-1 acceptance mask: ``True`` where ``residual <= eps``."""
    return np.asarray(residual) <= eps


# Backward-compatible alias -- existing experiment code calls
# `spurious_residual(eigs, psi_X, psi_Y, delta)`.  It IS `resdmd_residual`
# (function-value form); `delta` is the pseudospectral tolerance `eps`.  Prefer
# `resdmd_residual` in new code.
spurious_residual = resdmd_residual


def resdmd_pseudospectrum(z_grid, G, A, L):
    r"""ResDMD pseudospectrum (Colbrook Algorithm 2): ``tau(z) = min_g res(z, g)``.

    For each complex ``z``, ``tau(z)`` is the square root of the smallest
    generalised eigenvalue of the Hermitian pencil ``(D(z), G)`` with
    ``D(z) = L - z A^* - conj(z) A + |z|^2 G``.  ``z`` lies in the
    ``epsilon``-pseudospectrum iff ``tau(z) <= epsilon`` -- the verified region
    within which a true eigenvalue must lie.  Accepts a scalar or array of ``z``.
    """
    from scipy.linalg import eigh

    z_grid = np.atleast_1d(z_grid)
    G_h = (G + G.conj().T) / 2
    tau = np.empty(z_grid.shape, dtype=float)
    for idx, z in np.ndenumerate(z_grid):
        D = L - z * A.conj().T - np.conj(z) * A + np.abs(z) ** 2 * G
        D = (D + D.conj().T) / 2
        ev = eigh(D, G_h, eigvals_only=True)
        tau[idx] = np.sqrt(max(float(ev.min()), 0.0))
    return tau if tau.shape != (1,) else float(tau[0])


# ===========================================================
# --- compilation function for analysing spectral metrics ---
# ===========================================================


def analyse_spectrum(
    modes_records,
    trials_records,
    out_prefix,
    *,
    compute_extra_axes=True,
    horizon_H=H,
    horizon_mode_weights=MODE_W,
):
    modes_df = pd.DataFrame(modes_records).copy()
    trials_df = pd.DataFrame(trials_records).copy()

    if "spectral_gap" not in modes_df.columns:
        raise ValueError(
            f"modes_df is missing 'spectral_gap'. Columns: {modes_df.columns.tolist()}"
        )

    summary = modes_df.groupby(
        ["kernel", "kind", "method", "eigenfunction_id"], as_index=False
    ).agg(
        n=("spectral_bias", "size"),
        bias_mean=("spectral_bias", "mean"),
        dist_mean=("metric_distortion", "mean"),
        spurious_mean=("residual_spurious_score", "mean"),
        spurious_std=("residual_spurious_score", "std"),
    )

    rows = []
    for (kernel, kind, method), g in modes_df.groupby(["kernel", "kind", "method"]):
        gg = g[["spectral_bias", "spectral_gap"]].dropna()
        corr = gg["spectral_bias"].corr(gg["spectral_gap"]) if len(gg) > 1 else np.nan
        rows.append(
            {
                "kernel": kernel,
                "kind": kind,
                "method": method,
                "bias_gap_corr": corr,
            }
        )
    corr_df = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # Extended grand-score axes (candidate-level, optional)
    # ------------------------------------------------------------
    axes_df = _extended_axes(
        modes_df,
        trials_df,
        compute_extra_axes=compute_extra_axes,
        horizon_H=horizon_H,
        horizon_mode_weights=horizon_mode_weights,
    )

    summary.to_csv(f"{out_prefix}_summary.csv", index=False)
    modes_df.to_csv(f"{out_prefix}_metrics.csv", index=False)
    trials_df.to_csv(f"{out_prefix}_trials.csv", index=False)
    corr_df.to_csv(f"{out_prefix}_corr.csv", index=False)
    axes_df.to_csv(f"{out_prefix}_axes.csv", index=False)

    # ------------------------------------------------------------
    # Scatter diagnostics with compact, deduplicated legends
    # ------------------------------------------------------------
    def _short_label(x, max_chars=24):
        s = str(x)
        m = re.search(r"([A-Za-z_][A-Za-z0-9_]*\s*=\s*[^,)]+)", s)
        if m:
            return m.group(1).replace(" ", "")
        return s if len(s) <= max_chars else s[: max_chars - 1] + "…"

    def _build_group_label(kernel, kind, method, include_kernel=False):
        """
        Keep legend labels compact.
        By default, legend shows only kind + method.
        Set include_kernel=True only if the number of groups is small.
        """
        if include_kernel:
            return f"{_short_label(kernel)}, {kind} / {method}"
        return f"{kind} / {method}"

    def _dedup_legend(
        ax, title=None, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8
    ):
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l and l != "_nolegend_" and l not in seen:
                seen[l] = h
        if seen:
            ax.legend(
                seen.values(),
                seen.keys(),
                frameon=False,
                fontsize=fontsize,
                title=title,
                loc=loc,
                bbox_to_anchor=bbox_to_anchor,
                borderaxespad=0.0,
            )

    # -----------------------------
    # Figure 1: spectral bias vs spectral gap
    # -----------------------------
    fig1, ax = plt.subplots(figsize=(8.5, 4.8))
    # Set this to True only when there are very few groups
    include_kernel_in_legend = True
    for (kernel, kind, method), g in modes_df.groupby(["kernel", "kind", "method"]):
        g = g[np.isfinite(g["spectral_bias"]) & np.isfinite(g["spectral_gap"])].copy()
        if g.empty:
            continue
        label = _build_group_label(
            kernel=kernel,
            kind=kind,
            method=method,
            include_kernel=include_kernel_in_legend,
        )
        ax.scatter(
            g["spectral_bias"],
            g["spectral_gap"],
            s=20,
            alpha=0.7,
            label=label,
        )
    ax.set_xlabel("Spectral bias")
    ax.set_ylabel("Spectral gap")
    ax.set_title("Spectral bias vs Spectral gap")
    ax.grid(alpha=0.25)
    _dedup_legend(
        ax,
        title="Condition / method",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
    )
    fig1.tight_layout()
    fig1.savefig(f"{out_prefix}_gap_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # -----------------------------
    # Figure 2: spectral bias vs residual spurious score
    # -----------------------------
    fig2, ax = plt.subplots(figsize=(8.5, 4.8))
    for (kernel, kind, method), g in modes_df.groupby(["kernel", "kind", "method"]):
        g = g[
            np.isfinite(g["spectral_bias"]) & np.isfinite(g["residual_spurious_score"])
        ].copy()
        if g.empty:
            continue
        label = _build_group_label(
            kernel=kernel,
            kind=kind,
            method=method,
            include_kernel=include_kernel_in_legend,
        )
        ax.scatter(
            g["spectral_bias"],
            g["residual_spurious_score"],
            s=20,
            alpha=0.7,
            label=label,
        )
    ax.set_xlabel("Spectral bias")
    ax.set_ylabel("Residual spurious score")
    ax.set_title("Spectral bias vs Residual spurious score")
    ax.grid(alpha=0.25)
    _dedup_legend(
        ax,
        title="Condition / method",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
    )
    fig2.tight_layout()
    fig2.savefig(f"{out_prefix}_spurious_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    # NOTE: return arity extended by one -- `axes_df` sits next to summary/corr_df.
    return modes_df, trials_df, summary, corr_df, axes_df, fig1, fig2


def _extended_axes(
    modes_df,
    trials_df,
    *,
    compute_extra_axes=True,
    horizon_H=H,
    horizon_mode_weights=MODE_W,
):
    """Candidate-level table of the extended axes; empty (keys only) if none apply.

    - horizon terms: computed from est_eig_real/est_eig_imag in `modes_df`
    - sc / vamp2:    averaged over trials from `trials_df` (whichever exist)

    Returned columns: kernel, kind, method [, agg_horizon_instab,
    agg_horizon_sens, sc, vamp2].  Feed directly to
    kernel_spectral_score(..., extra_metrics=axes_df).
    """
    keys = ["kernel", "kind", "method"]
    base = (
        modes_df[keys].drop_duplicates().reset_index(drop=True)
        if set(keys).issubset(modes_df.columns)
        else pd.DataFrame(columns=keys)
    )
    if not compute_extra_axes or base.empty:
        return base

    parts = [base.set_index(keys)]

    # --- long-horizon terms (need per-trial est_eig columns) ---------------
    if {"est_eig_real", "est_eig_imag"}.issubset(modes_df.columns):
        hz = horizon_terms(
            modes_df, H=horizon_H, mode_weights=horizon_mode_weights, group_cols=keys
        )
        if len(hz):
            parts.append(hz.set_index(keys))

    # --- SSL criteria (mean over trials / data draws) ----------------------
    ssl_cols = [c for c in ("sc", "vamp2") if c in trials_df.columns]
    tkeys = [c for c in keys if c in trials_df.columns]
    if ssl_cols and tkeys:
        ssl = trials_df.groupby(tkeys, as_index=False)[ssl_cols].mean()
        parts.append(ssl.set_index(tkeys))

    axes_df = pd.concat(parts, axis=1).reset_index()
    return axes_df
