import numpy as np
from typing import Dict, Optional
from nmrkit.core import NMRData, LinearGenerator
from nmrkit.utils import (
    validate_dimension,
    update_dimension_info,
    update_domain_metadata,
    validate_param_type,
)


def fourier_transform(
    data: NMRData,
    dim: int = 0,
    inverse: bool = False,
    shift: Optional[bool] = None,
    fft_sign: Optional[int] = None,
) -> NMRData:
    """Apply Fourier Transform to a specific dimension of NMR data.

    Args:
        data: NMRData object to process
        dim: Dimension index to apply FT to (default: 0)
        inverse: If True, apply inverse FT (default: False)
        shift: If True, shift the zero-frequency component to the center.
               If False, do not shift. If None, use shift=True for forward FT
               and shift=False for inverse FT (default: None)
        fft_sign: Fourier sign convention for forward transforms. Use 1 for
               np.fft.fftn and -1 for the opposite convention. If None, use
               the dimension metadata value "fft_sign", defaulting to 1.

    Returns:
        NMRData: New NMRData object with FT applied
    """
    # Validate dimension
    validate_dimension(data, dim)

    # Validate parameters
    validate_param_type("shift", shift, (bool, type(None)))
    validate_param_type("fft_sign", fft_sign, (int, type(None)))

    # Create a copy to avoid modifying original data
    result = data.copy()

    # Check if dimension is Fourier transform capable
    if not result.dimensions[dim].can_ft:
        raise ValueError(f"Dimension {dim} is not Fourier transform capable")

    if fft_sign is None:
        fft_sign = int(result.dimensions[dim].domain_metadata.get("fft_sign", 1))

    if fft_sign not in {1, -1}:
        raise ValueError(f"fft_sign must be 1 or -1, got {fft_sign}")

    # Determine domain type after transformation
    if inverse:
        new_domain = "time"
        fft_func = np.fft.ifftn
        scale_factor = result.dimensions[dim].size
    else:
        new_domain = "frequency"
        fft_func = np.fft.fftn if fft_sign == 1 else np.fft.ifftn
        scale_factor = 1.0

    # Apply Fourier Transform
    result.data = fft_func(result.data, axes=(dim,)) / scale_factor
    if not inverse and fft_sign == -1:
        # np.fft.ifftn includes 1/N normalization. Match forward FFT scaling
        # when it is used only to select the opposite sign convention.
        result.data = result.data * result.dimensions[dim].size

    # Determine shift behavior if not explicitly specified
    if shift is None:
        shift = not inverse  # Default: shift for forward FT, no shift for inverse FT

    # Apply FT shift if requested
    if shift:
        result.data = np.fft.fftshift(result.data, axes=(dim,))

    # Calculate new axis parameters based on domain transformation
    old_dim = result.dimensions[dim]
    new_axis_generator = old_dim.axis_generator
    new_unit = old_dim.unit

    if not inverse:  # Forward FT: time → frequency
        if isinstance(new_axis_generator, LinearGenerator):
            time_increment = new_axis_generator.step
            freq_increment = 1.0 / (old_dim.size * time_increment)

            if shift:
                # If transmitter_offset is not set, use 0.0 as default
                freq_start = (
                    old_dim.transmitter_offset
                    if old_dim.transmitter_offset is not None
                    else 0.0
                ) - (old_dim.size * freq_increment) / 2.0
            else:
                freq_start = 0.0

            new_axis_generator = LinearGenerator(start=freq_start, step=freq_increment)

        # Set appropriate unit for frequency domain
        new_unit = "Hz"
    else:  # Inverse FT: frequency → time
        # For inverse FT, we need to calculate time axis parameters
        if isinstance(new_axis_generator, LinearGenerator):
            # For inverse FT, the time increment (Δt) is 1/(size * Δf)
            freq_increment = new_axis_generator.step
            time_increment = 1.0 / (old_dim.size * freq_increment)

            # Start at zero for time domain
            new_axis_generator = LinearGenerator(start=0.0, step=time_increment)

        # Set appropriate unit for time domain
        new_unit = "s"

    # Update dimension information
    result.dimensions[dim] = update_dimension_info(
        result.dimensions[dim],
        domain_type=new_domain,
        unit=new_unit,
        axis_generator=new_axis_generator,
        can_ft=True,  # Both time and frequency domains should be FT capable
    )

    # Update domain metadata
    update_domain_metadata(
        result,
        dim,
        ft_applied=True,
        ft_inverse=inverse,
        ft_shifted=shift,
        fft_sign=fft_sign,
    )

    return result


def ft_shift(data: NMRData, dim: int = 0, shift: bool = True) -> NMRData:
    """Shift the zero-frequency component of the spectrum.

    Args:
        data: NMRData object to process
        dim: Dimension index to apply shift to (default: 0)
        shift: If True, shift zero-frequency to center (fftshift).
               If False, shift zero-frequency to beginning (ifftshift) (default: True)

    Returns:
        NMRData: New NMRData object with shift applied
    """
    # Validate dimension
    validate_dimension(data, dim)

    # Create a copy to avoid modifying original data
    result = data.copy()

    # Apply appropriate shift
    if shift:
        result.data = np.fft.fftshift(result.data, axes=(dim,))
    else:
        result.data = np.fft.ifftshift(result.data, axes=(dim,))

    # Update domain metadata
    update_domain_metadata(result, dim, ft_shifted=shift)

    return result


# For backward compatibility
def ft_unshift(data: NMRData, dim: int = 0) -> NMRData:
    """Undo the FT shift (shift zero-frequency component back to the beginning).

    This is equivalent to ft_shift(data, dim, shift=False).

    Args:
        data: NMRData object to process
        dim: Dimension index to apply unshift to (default: 0)

    Returns:
        NMRData: New NMRData object with FT unshift applied
    """
    return ft_shift(data, dim, shift=False)
