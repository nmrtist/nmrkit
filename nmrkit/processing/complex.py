"""Complex number processing functionality for nmrkit."""

import numpy as np
from nmrkit.core.data import NMRData
from nmrkit.utils import update_dimension_info
from nmrkit.utils.complex import complexify


def _infer_indirect_complexification(data: NMRData, dim: int = 1):
    """Infer indirect-dimension quadrature storage from reader metadata."""
    dim_info = data.dimensions[dim]
    encoding = dim_info.domain_metadata.get("complex_pair_encoding")

    if encoding == "none":
        return None
    if encoding in {"interleaved", "separated"}:
        return encoding, dim_info.domain_metadata.get("first_component", "real")

    if data.source_format == "topspin":
        fnmode = dim_info.domain_metadata.get("fnmode")
        if fnmode == 1:
            return None
        if fnmode in {2, 4, 5, 6}:
            return "interleaved", "real"

    if data.source_format == "delta":
        logical_size = dim_info.domain_metadata.get("logical_size")
        if logical_size is not None and dim_info.size == 2 * int(logical_size):
            return "separated", "real"

    return None


def complexify_indirect_dim(
    data: NMRData, mode: str = "interleaved", first_component: str = "real"
) -> NMRData:
    """Complexify the indirect dimension of 2D NMR data.

    This function processes the indirect dimension by discarding the direct
    dimension imaginary channel, then re-complexifying the real channel using
    the requested or inferred indirect quadrature storage.

    Parameters
    ----------
    data : NMRData
        Input NMR data.
    mode : str, optional
        How real and imaginary parts are stored in the input data:
            - 'auto': Infer storage from dimension metadata
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

    if data.ndim < 2:
        raise ValueError("Indirect complexification requires at least 2D data")

    if mode == "auto":
        inferred = _infer_indirect_complexification(data)
        if inferred is None:
            data.dimensions[1].domain_metadata["complexified_indirect"] = False
            return data
        mode, first_component = inferred

    # Extract real parts only
    real_data = np.real(data.data)

    # Complexify the data using the specified mode
    complex_data = complexify(real_data, mode=mode, first_component=first_component)

    # Update the data array
    data.data = complex_data

    # Update the dimension size
    old_dim = data.dimensions[1]
    metadata = old_dim.domain_metadata.copy()
    metadata.update(
        {
            "complexified_indirect": True,
            "complexified_mode": mode,
            "complexified_first_component": first_component,
            "storage_size": old_dim.size,
        }
    )
    data.dimensions[1] = update_dimension_info(
        old_dim,
        size=complex_data.shape[1],
        is_complex=True,
        domain_metadata=metadata,
    )

    return data
