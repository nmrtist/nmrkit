import numpy as np
from typing import Optional
from nmrkit.core import NMRData, DimensionInfo


def zero_fill(
    data: NMRData, dim: int = 0, size: Optional[int] = None, power_of_two: bool = True
) -> NMRData:
    """Zero fill a specific dimension of NMR data.

    Args:
        data: NMRData object to process
        dim: Dimension index to zero fill (default: 0)
        size: Target size after zero filling. If None, use next power of two.
        power_of_two: If True and size is None, zero fill to next power of two.

    Returns:
        NMRData: New NMRData object with zero filling applied
    """
    if dim < 0 or dim >= data.ndim:
        raise ValueError(f"Dimension index {dim} out of range for {
                data.ndim}D data")

    # Create a copy to avoid modifying original data
    result = data.copy()

    current_size = result.dimensions[dim].size

    # Determine target size
    if size is None:
        if power_of_two:
            # Find next power of two greater than current size
            target_size = 1 << (current_size - 1).bit_length()
        else:

            return result
    else:
        target_size = size

    if target_size < current_size:
        raise ValueError(
            f"Target size {target_size} must be greater than "
            f"current size {current_size}"
        )

    zeros_to_add = target_size - current_size

    if zeros_to_add == 0:
        return result

    # Create padding configuration
    pad_width = [(0, 0)] * result.ndim
    pad_width[dim] = (0, zeros_to_add)

    result.data = np.pad(result.data, pad_width, mode="constant", constant_values=0)

    # Update dimension information
    result.dimensions[dim] = DimensionInfo(
        size=target_size,
        is_complex=result.dimensions[dim].is_complex,
        spectral_width=result.dimensions[dim].spectral_width,
        observation_frequency=result.dimensions[dim].observation_frequency,
        nucleus=result.dimensions[dim].nucleus,
        domain_type=result.dimensions[dim].domain_type,
        can_ft=result.dimensions[dim].can_ft,
        unit=result.dimensions[dim].unit,
        transmitter_offset=result.dimensions[dim].transmitter_offset,
        axis_generator=result.dimensions[dim].axis_generator,
        domain_metadata=result.dimensions[dim].domain_metadata.copy(),
    )

    # Update domain metadata
    result.dimensions[dim].domain_metadata["zero_filled"] = True
    result.dimensions[dim].domain_metadata["original_size"] = current_size
    result.dimensions[dim].domain_metadata["target_size"] = target_size

    return result


def extract_region(
    data: NMRData, dim: int = 0, start: int = 0, end: Optional[int] = None
) -> NMRData:
    """Extract a region from a specific dimension of NMR data.

    Args:
        data: NMRData object to process
        dim: Dimension index to extract from (default: 0)
        start: Starting index (inclusive)
        end: Ending index (exclusive). If None, extract to the end.

    Returns:
        NMRData: New NMRData object with extracted region
    """
    if dim < 0 or dim >= data.ndim:
        raise ValueError(f"Dimension index {dim} out of range for {
                data.ndim}D data")

    # Create a copy to avoid modifying original data
    result = data.copy()

    current_size = result.dimensions[dim].size

    # Validate start and end indices
    if start < 0:
        raise ValueError(f"Start index {start} must be non-negative")

    if end is None:
        end = current_size

    if end > current_size:
        raise ValueError(f"End index {end} cannot exceed current size {current_size}")

    if start >= end:
        raise ValueError(f"Start index {start} must be less than end index {end}")

    # Create slice object
    slices = [slice(None)] * result.ndim
    slices[dim] = slice(start, end)

    result.data = result.data[tuple(slices)]

    # Update dimension information
    new_size = end - start
    result.dimensions[dim] = DimensionInfo(
        size=new_size,
        is_complex=result.dimensions[dim].is_complex,
        spectral_width=result.dimensions[dim].spectral_width,
        observation_frequency=result.dimensions[dim].observation_frequency,
        nucleus=result.dimensions[dim].nucleus,
        domain_type=result.dimensions[dim].domain_type,
        can_ft=result.dimensions[dim].can_ft,
        unit=result.dimensions[dim].unit,
        transmitter_offset=result.dimensions[dim].transmitter_offset,
        axis_generator=result.dimensions[dim].axis_generator,
        domain_metadata=result.dimensions[dim].domain_metadata.copy(),
    )

    # Update domain metadata
    result.dimensions[dim].domain_metadata["extracted"] = True
    result.dimensions[dim].domain_metadata["extraction_start"] = start
    result.dimensions[dim].domain_metadata["extraction_end"] = end

    return result
