"""Static-Frozen Causal Driver Innovation for mechanical forecasting."""

__version__ = "1.0.0"

from .data import MechanicalSeries, SeriesWindowDataset, Standardizer
from .discovery import discover_lagged_drivers

__all__ = [
    "MechanicalSeries",
    "SeriesWindowDataset",
    "Standardizer",
    "discover_lagged_drivers",
    "__version__",
]
