from importlib.metadata import PackageNotFoundError, version

from kooplearn.metrics import (
    directed_hausdorff_distance,
    metric_distortion,
    operator_norm_error,
    spectral_bias,
    spurious_eigenvalues,
)
from kooplearn.structs import DynamicalModes

try:
    __version__ = version("kooplearn")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "DynamicalModes",
    "__version__",
    "directed_hausdorff_distance",
    "metric_distortion",
    "operator_norm_error",
    "spectral_bias",
    "spurious_eigenvalues",
]
