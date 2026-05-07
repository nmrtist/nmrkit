import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from nmrkit.core import NMRData, DimensionInfo


def validate_dimension(data: NMRData, dim: int) -> None:
    """Validate that dimension index is within valid range.

    Args:
        data: NMRData object to validate against
        dim: Dimension index to validate

    Raises:
        ValueError: If dimension index is out of range
    """
    if dim < 0 or dim >= data.ndim:
        raise ValueError(f"Dimension index {dim} out of range for {
                data.ndim}D data")


def create_dimension_shape(ndim: int, dim: int, dim_size: int) -> List[int]:
    """Create a shape tuple for a dimension-specific array.

    Args:
        ndim: Total number of dimensions
        dim: Target dimension index
        dim_size: Size of the target dimension

    Returns:
        List[int]: Shape list with 1s except for the target dimension
    """
    shape = [1] * ndim
    shape[dim] = dim_size
    return shape


def update_dimension_info(dim_info: DimensionInfo, **kwargs: Any) -> DimensionInfo:
    """Update DimensionInfo object with new values.

    Args:
        dim_info: Original DimensionInfo object
        **kwargs: New values to update

    Returns:
        DimensionInfo: New DimensionInfo object with updated values
    """
    attributes = {
        "size": dim_info.size,
        "is_complex": dim_info.is_complex,
        "spectral_width": dim_info.spectral_width,
        "observation_frequency": dim_info.observation_frequency,
        "nucleus": dim_info.nucleus,
        "domain_type": dim_info.domain_type,
        "can_ft": dim_info.can_ft,
        "unit": dim_info.unit,
        "transmitter_offset": dim_info.transmitter_offset,
        "axis_generator": dim_info.axis_generator,
        "domain_metadata": dim_info.domain_metadata.copy(),
    }

    attributes.update(kwargs)

    return DimensionInfo(**attributes)


def update_domain_metadata(data: NMRData, dim: int, **kwargs: Any) -> None:
    """Update domain metadata for a specific dimension.

    Args:
        data: NMRData object to update
        dim: Dimension index to update
        **kwargs: Key-value pairs to add/update in domain_metadata
    """
    for key, value in kwargs.items():
        data.dimensions[dim].domain_metadata[key] = value


def get_time_array(dim_info: DimensionInfo) -> np.ndarray:
    """Generate time array from dimension information.

    Args:
        dim_info: DimensionInfo object containing axis generation information

    Returns:
        np.ndarray: Time array in seconds
    """
    return dim_info.generate_axis()


def validate_param_value(
    param_name: str,
    param_value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
) -> None:
    """Validate that a parameter value is within the specified range.

    Args:
        param_name: Name of the parameter for error messages
        param_value: Value of the parameter to validate
        min_value: Minimum allowed value (inclusive, default: None)
        max_value: Maximum allowed value (inclusive, default: None)

    Raises:
        ValueError: If parameter value is outside the specified range
    """
    if min_value is not None and param_value < min_value:
        raise ValueError(f"{param_name} must be >= {min_value}, got {param_value}")
    if max_value is not None and param_value > max_value:
        raise ValueError(f"{param_name} must be <= {max_value}, got {param_value}")


def validate_param_type(
    param_name: str, param_value: Any, expected_types: tuple
) -> None:
    """Validate that a parameter is of the expected type.

    Args:
        param_name: Name of the parameter for error messages
        param_value: Value of the parameter to validate
        expected_types: Tuple of expected types

    Raises:
        TypeError: If parameter is not of the expected type
    """
    # Special case: boolean is a subclass of int, but we want to reject it for
    # int/float parameters
    if isinstance(param_value, bool) and (
        int in expected_types or float in expected_types
    ):
        raise TypeError(f"{param_name} must be of type {expected_types}, got boolean")
    if not isinstance(param_value, expected_types):
        raise TypeError(f"{param_name} must be of type {expected_types}, got {
                type(param_value).__name__}")


def validate_param_options(
    param_name: str, param_value: Any, allowed_options: list
) -> None:
    """Validate that a parameter value is in the allowed options list.

    Args:
        param_name: Name of the parameter for error messages
        param_value: Value of the parameter to validate
        allowed_options: List of allowed values

    Raises:
        ValueError: If parameter value is not in the allowed options list
    """
    if param_value not in allowed_options:
        raise ValueError(
            f"{param_name} must be one of {allowed_options}, got {param_value}"
        )
