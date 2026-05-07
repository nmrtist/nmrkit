"""Complex number processing functionality for nmrkit."""

import numpy as np
from nmrkit.core.data import NMRData
from nmrkit.utils.complex import complexify


def complexify_indirect_dim(
    data: NMRData, mode: str = "interleaved", first_component: str = "real"
) -> NMRData:
    """Complexify the indirect dimension of 2D NMR data.

    This function processes the indirect dimension by:
    1. Discarding all imaginary parts
    2. Re-complexifying the real parts (interleaved) and updating parameters

    Parameters
    ----------
    data : NMRData
        Input NMR data.
    mode : str, optional
        How real and imaginary parts are stored in the input data:
            - 'interleaved': Real and imaginary parts are interleaved (e.g., [Re0, Im0, Re1, Im1, ...])
            - 'separated': All real parts followed by all imaginary parts (e.g., [Re0, Re1, ..., Im0, Im1, ...])
    first_component : str, optional
        Whether the first component in each pair is real or imaginary (only applicable for 'interleaved' mode):
            - 'real': First component is real, second is imaginary (default)
            - 'imaginary': First component is imaginary, second is real

    Returns
    -------
    NMRData
        Processed NMR data with complexified indirect dimension.
    """
    # Create a copy of the data to avoid modifying the original
    data = data.copy()

    # Extract real parts only
    real_data = np.real(data.data)

    # Complexify the data using the specified mode
    complex_data = complexify(real_data, mode=mode, first_component=first_component)

    # Update the data array
    data.data = complex_data

    # Update the dimension size
    data.dimensions[1].size = complex_data.shape[1]

    return data
