import numpy as np


def complexify(
    data: np.ndarray, mode: str = "interleaved", first_component: str = "real"
) -> np.ndarray:
    """Convert real-only data to complex data.

    Args:
        data: Input real-only data array.
        mode: How real and imaginary parts are stored in the input data:
            - 'interleaved': Real and imaginary parts are interleaved (e.g., [Re0, Im0, Re1, Im1, ...])
            - 'separated': All real parts followed by all imaginary parts (e.g., [Re0, Re1, ..., Im0, Im1, ...])
        first_component: Whether the first component in each pair is real or imaginary (only applicable for 'interleaved' mode):
            - 'real': First component is real, second is imaginary (default)
            - 'imaginary': First component is imaginary, second is real

    Returns:
        np.ndarray: Complex data array with real and imaginary parts properly combined.
    """
    if mode not in ["interleaved", "separated"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'interleaved' or 'separated'.")

    if first_component not in ["real", "imaginary"]:
        raise ValueError(
            f"Invalid first_component: {first_component}. Must be 'real' or 'imaginary'."
        )

    if mode == "interleaved":
        # Check if data size is even for interleaved mode
        if data.shape[-1] % 2 != 0:
            raise ValueError(
                f"Data size in the last dimension must be even for interleaved mode, got {data.shape[-1]}."
            )

        # Calculate new shape
        new_shape = list(data.shape)
        new_shape[-1] = new_shape[-1] // 2

        # Create complex array
        complex_data = np.empty(new_shape, dtype=np.complex128)

        if first_component == "real":
            # Real part first: [Re0, Im0, Re1, Im1, ...] -> [Re0+Im0j,
            # Re1+Im1j, ...]
            complex_data.real = data[..., ::2]
            complex_data.imag = data[..., 1::2]
        else:
            # Imaginary part first: [Im0, Re0, Im1, Re1, ...] -> [Re0+Im0j,
            # Re1+Im1j, ...]
            complex_data.imag = data[..., ::2]
            complex_data.real = data[..., 1::2]

    else:  # mode == 'separated'
        # Check if data size is even for separated mode
        if data.shape[-1] % 2 != 0:
            raise ValueError(
                f"Data size in the last dimension must be even for separated mode, got {data.shape[-1]}."
            )

        # Calculate new shape
        new_shape = list(data.shape)
        new_shape[-1] = new_shape[-1] // 2

        # Create complex array
        complex_data = np.empty(new_shape, dtype=np.complex128)

        half_size = data.shape[-1] // 2
        if first_component == "real":
            # Real parts first: [Re0, Re1, ..., Im0, Im1, ...] -> [Re0+Im0j,
            # Re1+Im1j, ...]
            complex_data.real = data[..., :half_size]
            complex_data.imag = data[..., half_size:]
        else:
            # Imaginary parts first: [Im0, Im1, ..., Re0, Re1, ...] ->
            # [Re0+Im0j, Re1+Im1j, ...]
            complex_data.imag = data[..., :half_size]
            complex_data.real = data[..., half_size:]

    return complex_data


def decomplexify(
    data: np.ndarray, mode: str = "interleaved", first_component: str = "real"
) -> np.ndarray:
    """Convert complex data to real-only data (inverse of complexify).

    Args:
        data: Input complex data array.
        mode: How real and imaginary parts are stored in the output data:
            - 'interleaved': Real and imaginary parts are interleaved (e.g., [Re0, Im0, Re1, Im1, ...])
            - 'separated': All real parts followed by all imaginary parts (e.g., [Re0, Re1, ..., Im0, Im1, ...])
        first_component: Whether the first component in each pair is real or imaginary (only applicable for 'interleaved' mode):
            - 'real': First component is real, second is imaginary (default)
            - 'imaginary': First component is imaginary, second is real

    Returns:
        np.ndarray: Real-only data array with real and imaginary parts properly stored.
    """
    if mode not in ["interleaved", "separated"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'interleaved' or 'separated'.")

    if first_component not in ["real", "imaginary"]:
        raise ValueError(
            f"Invalid first_component: {first_component}. Must be 'real' or 'imaginary'."
        )

    # Calculate new shape
    new_shape = list(data.shape)
    new_shape[-1] = new_shape[-1] * 2

    # Create real array
    real_data = np.empty(new_shape, dtype=np.float64)

    if mode == "interleaved":
        if first_component == "real":
            # Real part first: [Re0+Im0j, Re1+Im1j, ...] -> [Re0, Im0, Re1,
            # Im1, ...]
            real_data[..., ::2] = data.real
            real_data[..., 1::2] = data.imag
        else:
            # Imaginary part first: [Re0+Im0j, Re1+Im1j, ...] -> [Im0, Re0,
            # Im1, Re1, ...]
            real_data[..., ::2] = data.imag
            real_data[..., 1::2] = data.real

    else:  # mode == 'separated'
        half_size = real_data.shape[-1] // 2
        if first_component == "real":
            # Real parts first: [Re0+Im0j, Re1+Im1j, ...] -> [Re0, Re1, ...,
            # Im0, Im1, ...]
            real_data[..., :half_size] = data.real
            real_data[..., half_size:] = data.imag
        else:
            # Imaginary parts first: [Re0+Im0j, Re1+Im1j, ...] -> [Im0, Im1,
            # ..., Re0, Re1, ...]
            real_data[..., :half_size] = data.imag
            real_data[..., half_size:] = data.real

    return real_data
