# tests/test_metrics.py

import numpy as np
import pytest

from kooplearn.metrics import (
    directed_hausdorff_distance,
    metric_distortion,
    operator_norm_error,
    spectral_bias,
    spurious_eigenvalue,
)


class TestDirectedHausdorffDistance:
    def test_known_example(self):
        pred = np.array([1.0, 5.0, 6.0])
        reference = np.array([2.0, 4.0, 7.0])

        result = directed_hausdorff_distance(pred, reference)

        assert result == pytest.approx(1.0)

    def test_1d_inputs(self):
        pred = np.array([[1.0, 2.0]])
        reference = np.array([1.0, 2.0])

        with pytest.raises(AssertionError):
            directed_hausdorff_distance(pred, reference)


class TestMetricDistortion:
    def test_diagonal_covariance(self):
        psi_hat = np.array([1.0, 0.0])
        c_hat = np.array([[4.0, 0.0], [0.0, 1.0]])

        result = metric_distortion(psi_hat, c_hat)

        # ||psi_hat|| = 1, <C psi_hat, psi_hat> = 4
        assert result == pytest.approx(0.5)

    def test_complex_vector(self):
        psi_hat = np.array([1.0 + 0.0j, 1.0j])
        c_hat = np.eye(2)

        result = metric_distortion(psi_hat, c_hat)

        assert result == pytest.approx(1.0)

    def test_1d_eigenfunction(self):
        psi_hat = np.eye(2)
        c_hat = np.eye(2)

        with pytest.raises(ValueError, match="1D"):
            metric_distortion(psi_hat, c_hat)

    def test_square_covariance(self):
        psi_hat = np.array([1.0, 0.0])
        c_hat = np.ones((2, 3))

        with pytest.raises(ValueError, match="square"):
            metric_distortion(psi_hat, c_hat)

    def test_compatible_shapes(self):
        psi_hat = np.array([1.0, 0.0, 0.0])
        c_hat = np.eye(2)

        with pytest.raises(ValueError, match="incompatible"):
            metric_distortion(psi_hat, c_hat)

    def test_rejects_nonpositive(self):
        psi_hat = np.array([1.0, 0.0])
        c_hat = np.zeros((2, 2))

        with pytest.raises(ValueError, match="strictly positive"):
            metric_distortion(psi_hat, c_hat)


class TestOperatorNormError:
    def test_known_matrix_difference(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        a_hat = np.array([[3.0, 0.0], [0.0, 1.0]])

        result = operator_norm_error(a, a_hat)

        # difference is [[2, 0], [0, 0]], spectral norm = 2
        assert result == pytest.approx(2.0)

    def test_2d_inputs(self):
        a = np.array([1.0, 2.0])
        a_hat = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="2D"):
            operator_norm_error(a, a_hat)

    def test_same_shape(self):
        a = np.eye(2)
        a_hat = np.eye(3)

        with pytest.raises(ValueError, match="same shape"):
            operator_norm_error(a, a_hat)


class TestSpectralBias:
    def test_product_definition(self):
        result = spectral_bias(metric_distortion=2.5, truncation_term=0.2)
        assert result == pytest.approx(0.5)

    def test_zero_values(self):
        assert spectral_bias(metric_distortion=0.0, truncation_term=1.0) == pytest.approx(0.0)
        assert spectral_bias(metric_distortion=2.0, truncation_term=0.0) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("metric_distortion", "truncation_term"),
    [
        pytest.param(-1.0, 0.1, id="negative-metric-distortion"),
        pytest.param(1.0, -0.1, id="negative-truncation-term"),
    ],
)
def test_rejects_negative_inputs(self, metric_distortion, truncation):
    with pytest.raises(ValueError, match="non-negative"):
        spectral_bias(
            metric_distortion=metric_distortion,
            truncation=truncation,
        )


class TestSpuriousEigenvalue:
    def test_counts_spurious_eigenvalues(self):
        estimated = np.array([0.9 + 0.0j, 0.5 + 0.2j, -1.5 + 0.0j])
        reference = np.array([0.9 + 0.0j, 0.5 + 0.2j])

        result = spurious_eigenvalue(estimated, reference, delta=0.1)

        assert result == 1

    def test_no_spurious_eigenvalues(self):
        estimated = np.array([1.0, 2.0])
        reference = np.array([1.01, 1.99])

        result = spurious_eigenvalue(estimated, reference, delta=0.05)

        assert result == 0

    def test_positive_delta(self):
        estimated = np.array([1.0])
        reference = np.array([1.0])

        with pytest.raises(ValueError, match="strictly positive"):
            spurious_eigenvalue(estimated, reference, delta=0.0)

    def test_1d_inputs(self):
        estimated = np.array([[1.0]])
        reference = np.array([1.0])

        with pytest.raises(ValueError, match="1D"):
            spurious_eigenvalue(estimated, reference, delta=0.1)
