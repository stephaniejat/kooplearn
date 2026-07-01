# src/kooplearn/metrics.py
# This file has been modified from the original version
# to add three new diagnostic metrics from Kostic et al. 2023:
# - metric distortion
# - spectral bias
# - spurious eigenvalue count

import numpy as np


# This function is the same as the original version.
def directed_hausdorff_distance(pred: np.ndarray, reference: np.ndarray):
    """One-sided directed Hausdorff distance between two 1D sets. Useful for computing distances
    between estimated eigenvalues

    Calculates the directed Hausdorff distance
    :math:`\\vec{H}(A, B) = \\max_{a \\in A} \\min_{b \\in B} \\|a - b\\|_p` where :math:`A` is the
    set of points in "pred" and :math:`B` is the set of points in "reference". The current
    implementation uses the :math:`L_1` norm: :math:`\\|a - b\\|_1 = |a - b|`.

    Parameters
    ----------
    pred : numpy.ndarray
        The set of predicted points :math:`A`. Must be a 1D array.
    reference : numpy.ndarray
        The set of reference points :math:`B`. Must be a 1D array.

    Returns
    -------
    float
        The directed Hausdorff distance between "pred"and "reference".

    Raises
    ------
    AssertionError
        If "pred" or "reference" are not 1-dimensional arrays.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        from kooplearn.metrics import directed_hausdorff_distance
        pred = np.array([1, 5, 6])
        reference = np.array([2, 4, 7])
        directed_hausdorff_distance(pred, reference)
        # Will print np.float64(1.0)
    """
    pred = np.asanyarray(pred)
    reference = np.asanyarray(reference)
    assert pred.ndim == 1
    assert reference.ndim == 1

    distances = np.zeros((pred.shape[0], reference.shape[0]), dtype=np.float64)
    for pred_idx, pred_pt in enumerate(pred):
        for reference_idx, reference_pt in enumerate(reference):
            distances[pred_idx, reference_idx] = np.abs(pred_pt - reference_pt)
    hausdorff_dist = np.max(np.min(distances, axis=1))
    return hausdorff_dist


# Edits from Kostic et al. 2023 inserted below.
# ──────────────────────────────────────────────────────────────────
# 1.  Operator norm error
# ──────────────────────────────────────────────────────────────────


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

    Parameters
    ----------
    true_operator : numpy.ndarray
        Array representing the reference action :math:`A_\pi S`.
    estimated_operator : numpy.ndarray
        Array representing the estimated action :math:`S \widehat{G}`.
        Must have the same shape as "true_operator".

    Returns
    -------
    float
        The spectral norm of the discrepancy if the inputs are matrices, or the
        Euclidean norm if the inputs are vectors.

    Raises
    ------
    ValueError
        If the inputs do not have the same shape.

    Notes
    -----
    For 1D arrays, this function returns the Euclidean norm.
    For 2D arrays this function returns the spectral norm,
    the largest singular value of the difference.

    This function is most useful when a trusted reference operator is
    available, for example in synthetic experiments or benchmark problems.
    It should not be interpreted as automatically estimating the theoretical
    quantity :math:`\|A_\pi S - S\widehat{G}\|` from data alone.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        from kooplearn.metrics import operator_norm_error

        A_pi_S = np.array([[1.0, 0.0], [0.0, 0.8]])
        S_Ghat = np.array([[0.9, 0.1], [0.0, 0.75]])
        err = operator_norm_error(A_pi_S, S_Ghat)
        print(err)
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


# ──────────────────────────────────────────────────────────────────
# 2.  Metric distortion
# ──────────────────────────────────────────────────────────────────

def metric_distortion(eigenfunction: np.ndarray, covariance: np.ndarray):
    r"""Metric distortion of a vector under the observation map.

    Computes the empirical metric distortion:

    .. math::
        \widehat{\eta}_i :=
        \frac{\|\widehat{\psi}_i\|}{\sqrt{
        \langle \widehat{C}\widehat{\psi}_i, \widehat{\psi}_i \rangle
        }}.

    Parameters
    ----------
    eigenfunction : numpy.ndarray, shape (n_features,)
        Empirical eigenfunction :math:`\widehat{\psi}_i`.
    covariance : numpy.ndarray, shape (n_features, n_features)
        Empirical covariance operator :math:`\widehat{C}` represented as a matrix.

    Returns
    -------
    float
        The empirical metric distortion :math:`\widehat{\eta}_i`.

    Raises
    ------
    ValueError
        If "eigenfunction" is not one-dimensional.
    ValueError
        If "covariance" is not a square matrix compatible with "eigenfunction".
    ValueError
        If :math:`\langle \widehat{C}\widehat{\psi}_i, \widehat{\psi}_i \rangle \leq 0`.

    Notes
    -----
    This is a practical diagnostic rather than the theorem-level quantity.

    The denominator is computed as:
    :math:`\langle \widehat{C}\widehat{\psi}_i, \widehat{\psi}_i \rangle`,
    where :math:`\widehat{C}` is the empirical covariance operator
    and :math:`\widehat{\psi}_i` is the empirical eigenfunction.
    Only its real part is retained, which is appropriate when
    "covariance" is Hermitian up to numerical error.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        from kooplearn.metrics import metric_distortion

        psi_hat = np.array([1.0, 0.0])
        C_hat = np.array([[2.0, 0.0], [0.0, 1.0]])
        eta_hat = metric_distortion(psi_hat, C_hat)
        print(eta_hat)
    """
    eigenfunction = np.asanyarray(eigenfunction)
    covariance = np.asanyarray(covariance)

    if eigenfunction.ndim != 1:
        raise ValueError(f"eigenfunction must be 1D, got array with shape {eigenfunction.shape}.")
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            f"covariance must be a square 2D array, got array with shape {covariance.shape}."
        )
    if covariance.shape[0] != eigenfunction.shape[0]:
        raise ValueError(
            "covariance and eigenfunction have incompatible shapes: "
            f"{covariance.shape} and {eigenfunction.shape}."
        )

    numerator = np.linalg.norm(eigenfunction)
    quadratic_form = np.real(np.vdot(eigenfunction, covariance @ eigenfunction))

    if quadratic_form <= 0.0:
        raise ValueError("The quadratic form <C_hat psi_hat, psi_hat> must be strictly positive.")

    return float(numerator / np.sqrt(quadratic_form))


# ──────────────────────────────────────────────────────────────────
# 3.  Spectral bias
# ──────────────────────────────────────────────────────────────────

def spectral_bias(metric_distortion: float, truncation: float):
    r"""Empirical spectral bias for a Koopman estimator.

    Computes the empirical spectral bias

    .. math::
        \mathrm{bs}_i(\widehat{G}_{r,\gamma})
        := \widehat{\eta}_i \, \rho_{r+1}(\widehat{G}_{r,\gamma}),

    where :math:`\widehat{\eta}_i` is the metric distortion and
    :math:`\rho_{r+1}(\widehat{G}_{r,\gamma})` is the estimator-specific
    truncation quantity controlling the rank-\((r+1)\) contribution.

    Parameters
    ----------
    metric_distortion : float
        Empirical metric distortion :math:`\widehat{\eta}_i`.
    truncation : float
        Estimator-specific quantity
        :math:`\rho_{r+1}(\widehat{G}_{r,\gamma})`.

    Returns
    -------
    float
        The empirical spectral bias
        :math:`\mathrm{bs}_i(\widehat{G}_{r,\gamma})`.

    Raises
    ------
    ValueError
        If ``metric_distortion`` is negative.
    ValueError
        If ``truncation`` is negative.

    Notes
    -----
    This function is intentionally algorithm-agnostic. For example, in the
    formulas of Kostic et al. (2023), one may take

    - :math:`\rho_{r+1}(\widehat{G}^{\mathrm{PCR}}_{r,\gamma})
      = \sigma_{r+1}(\widehat{C})`,
    - :math:`\rho_{r+1}(\widehat{G}^{\mathrm{RRR}}_{r,\gamma})
      = \sigma_{r+1}(\widehat{C}^{-1/2}\widehat{T})`.

    Examples
    --------

    .. code-block:: python

        from kooplearn.metrics import spectral_bias

        eta_hat = 2.5
        rho_hat = 0.1
        spectral_bias(eta_hat, rho_hat)
    """
    if metric_distortion < 0.0:
        raise ValueError(f"metric_distortion must be non-negative, got {metric_distortion}.")
    if truncation < 0.0:
        raise ValueError(f"truncation must be non-negative, got {truncation}.")

    return float(metric_distortion * truncation)


# ──────────────────────────────────────────────────────────────────
# 4.  Spurious eigenvalue count
# ──────────────────────────────────────────────────────────────────


def spurious_eigenvalues(
    estimated_eigenvalues: np.ndarray,
    reference_eigenvalues: np.ndarray,
    delta: float,
):
    r"""Count estimated eigenvalues that are farther than "delta" from a reference set.

    Computes the number of potentially spurious eigenvalues,

    .. math::
        N_{\mathrm{spur}}(\delta)
        =
        \#\left\{
            \widehat{\lambda}_j :
            \operatorname{dist}(\widehat{\lambda}_j, \sigma(A)) > \delta
        \right\},

    where the reference set (i.e., the true spectrum) is supplied explicitly,
    via "reference_eigenvalues".

    Parameters
    ----------
    estimated_eigenvalues : numpy.ndarray
        One-dimensional array of estimated eigenvalues.
    reference_eigenvalues : numpy.ndarray
        One-dimensional array of trusted or reference eigenvalues.
    delta : float
        Distance threshold. Estimated eigenvalues whose distance from the
        reference set exceeds "delta" are counted as spurious.

    Returns
    -------
    int
        Number of estimated eigenvalues farther than "delta" from the
        reference set.

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional or if "delta" is not positive.

    Notes
    -----
    This is a practical diagnostic rather than a theorem-level quantity.
    It is most useful when a trusted reference spectrum is available,
    for example from an analytically solvable problem, a high-resolution
    computation, or a benchmark estimator.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        from kooplearn.metrics import spurious_eigenvalues

        estimated = np.array([0.9 + 0.0j, 0.5 + 0.2j, -1.5 + 0.0j])
        reference = np.array([0.9 + 0.0j, 0.5 + 0.2j])
        spurious_eigenvalues(estimated, reference, delta=0.1)
    """
    estimated_eigenvalues = np.asanyarray(estimated_eigenvalues)
    reference_eigenvalues = np.asanyarray(reference_eigenvalues)

    if estimated_eigenvalues.ndim != 1:
        raise ValueError(
            f"estimated_eigenvalues must be a 1D array, got shape {estimated_eigenvalues.shape}."
        )
    if reference_eigenvalues.ndim != 1:
        raise ValueError(
            f"reference_eigenvalues must be a 1D array, got shape {reference_eigenvalues.shape}."
        )
    if delta <= 0.0:
        raise ValueError(f"delta must be strictly positive, got {delta}.")

    # dist(λ̂_j, σ(A)) = min_k |λ̂_j - σ_k|  in ℂ  (L1 on complex plane = |·|)
    # Expected shape: (r, 1) - (1, s) → (r, s)
    distances = np.abs(
        estimated_eigenvalues[:, np.newaxis] - reference_eigenvalues[np.newaxis, :]
    )  # shape (r, s)
    min_distances = np.min(distances, axis=1)  # shape (r,)
    return int(np.sum(min_distances > delta))
