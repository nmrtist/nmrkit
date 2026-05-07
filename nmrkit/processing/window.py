import numpy as np
from typing import Dict, Optional, Union
from nmrkit.core import NMRData
from nmrkit.utils import (
    validate_dimension,
    create_dimension_shape,
    update_domain_metadata,
    get_time_array,
)


def _apply_window_base(
    data: NMRData, dim: int, window: np.ndarray, window_type: str, **kwargs
) -> NMRData:
    """Base function to apply a window to NMR data.

    Args:
        data: NMRData object to process
        dim: Dimension index to apply window to
        window: Window array to apply
        window_type: Type of window function
        **kwargs: Additional metadata to store

    Returns:
        NMRData: New NMRData object with window applied
    """
    validate_dimension(data, dim)
    result = data.copy()

    dim_size = result.dimensions[dim].size
    window_shape = create_dimension_shape(result.ndim, dim, dim_size)
    window_reshaped = window.reshape(window_shape)

    # Check for NaN values in window
    if np.isnan(window_reshaped).any():
        # If window contains NaN, use a default window (ones)
        window_reshaped = np.ones(window_shape, dtype=np.float64)

    # Check for NaN values in data
    if np.isnan(result.data).any():
        # If data contains NaN, replace with zeros
        result.data = np.nan_to_num(result.data)

    result.data = result.data * window_reshaped

    metadata = {"window_type": window_type}
    metadata.update(kwargs)
    update_domain_metadata(result, dim, **metadata)

    return result


def exponential(data: NMRData, dim: int = 0, lb: float = 1.0) -> NMRData:
    """Apply exponential window function to NMR data.

    Args:
        data: NMRData object to process
        dim: Dimension index to apply window to (default: 0)
        lb: Line broadening parameter in Hz. Positive for decaying exponential,
            negative for resolution enhancement.

    Returns:
        NMRData: New NMRData object with exponential window applied
    """
    validate_dimension(data, dim)
    t = get_time_array(data.dimensions[dim])
    window = np.exp(-np.pi * lb * t)
    return _apply_window_base(data, dim, window, "exponential", window_lb=lb)


def gaussian(
    data: NMRData, dim: int = 0, gf: float = 1.0, shift: float = 0.0
) -> NMRData:
    """Apply Gaussian window function to NMR data.

    Args:
        data: NMRData object to process
        dim: Dimension index to apply window to (default: 0)
        gf: Time constant in seconds defining the Gaussian function width
        shift: Shift in seconds to center the Gaussian function

    Returns:
        NMRData: New NMRData object with Gaussian window applied
    """
    validate_dimension(data, dim)
    t = get_time_array(data.dimensions[dim])
    window = np.exp(-(((t - shift) / gf) ** 2))
    return _apply_window_base(
        data, dim, window, "gaussian", window_gf=gf, window_shift=shift
    )


def sine(data: NMRData, dim: int = 0, sb: float = 1.0, shift: float = 0.0) -> NMRData:
    """Apply sinebell window function to NMR data.

    Args:
        data: NMRData object to process
        dim: Dimension index to apply window to (default: 0)
        sb: Time constant in seconds. Positive for sinebell, negative for squared sinebell.
        shift: Shift in seconds to adjust the origin of the sinebell function

    Returns:
        NMRData: New NMRData object with sinebell window applied
    """
    validate_dimension(data, dim)
    t = get_time_array(data.dimensions[dim])

    arg = (t - shift) * np.pi / (2 * abs(sb))

    if sb > 0:
        window = np.sin(arg)
        window_type = "sinebell"
    else:
        window = np.sin(arg) ** 2
        window_type = "sinebell2"

    return _apply_window_base(
        data, dim, window, window_type, window_sb=sb, window_shift=shift
    )


def cosine(data: NMRData, dim: int = 0, squared: bool = False) -> NMRData:
    """Apply cosine window function to NMR data.

    Args:
        data: NMRData object to process
        dim: Dimension index to apply window to (default: 0)
        squared: Whether to apply squared cosine window

    Returns:
        NMRData: New NMRData object with cosine window applied
    """
    validate_dimension(data, dim)
    dim_size = data.dimensions[dim].size
    indices = np.arange(dim_size)

    if squared:
        window = np.cos(np.pi * ((indices + 0.5) / dim_size - 0.5)) ** 2
        window_type = "cosine2"
    else:
        window = np.cos(np.pi * ((indices + 0.5) / dim_size - 0.5))
        window_type = "cosine"

    return _apply_window_base(data, dim, window, window_type, window_squared=squared)


def trapezoidal(data: NMRData, dim: int = 0) -> NMRData:
    """Apply trapezoidal window function to NMR data.

    Args:
        data: NMRData object to process
        dim: Dimension index to apply window to (default: 0)

    Returns:
        NMRData: New NMRData object with trapezoidal window applied
    """
    validate_dimension(data, dim)
    dim_size = data.dimensions[dim].size
    window = np.ones(dim_size)
    return _apply_window_base(data, dim, window, "trapezoidal")


def first_point_scaling(data: NMRData, dim: int = 0, factor: float = 0.5) -> NMRData:
    """Apply first point scaling to NMR data.

    Args:
        data: NMRData object to process
        dim: Dimension index to apply scaling to (default: 0)
        factor: Scaling factor for the first point

    Returns:
        NMRData: New NMRData object with first point scaled
    """
    validate_dimension(data, dim)
    result = data.copy()

    idx = [slice(None)] * result.ndim
    idx[dim] = 0

    result.data[tuple(idx)] *= factor

    update_domain_metadata(
        result, dim, first_point_scaled=True, first_point_factor=factor
    )

    return result
