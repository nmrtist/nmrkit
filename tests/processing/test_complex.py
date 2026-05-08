import numpy as np

from nmrkit.core.data import DimensionInfo, LinearGenerator, NMRData
from nmrkit.processing.complex import complexify_indirect_dim


def test_complexify_indirect_auto_skips_qf_topspin_data():
    data_array = np.random.rand(8, 6) + 1j * np.random.rand(8, 6)
    dims = [
        DimensionInfo(size=8, is_complex=True, domain_type="frequency"),
        DimensionInfo(
            size=6,
            is_complex=False,
            domain_type="time",
            can_ft=True,
            axis_generator=LinearGenerator(start=0.0, step=0.001),
            domain_metadata={
                "fnmode": 1,
                "fnmode_name": "qf",
                "complex_pair_encoding": "none",
            },
        ),
    ]
    nmr_data = NMRData(data=data_array, dimensions=dims, source_format="topspin")

    result = complexify_indirect_dim(nmr_data, mode="auto")

    assert result.shape == (8, 6)
    np.testing.assert_array_equal(result.data, data_array)
    assert result.dimensions[1].domain_metadata["complexified_indirect"] is False


def test_complexify_indirect_auto_uses_delta_separated_pairs():
    real = np.arange(12, dtype=float).reshape(3, 4)
    imag = real + 100.0
    data_array = np.concatenate([real, imag], axis=1)
    dims = [
        DimensionInfo(size=3, is_complex=True, domain_type="frequency"),
        DimensionInfo(
            size=8,
            is_complex=True,
            domain_type="time",
            can_ft=True,
            axis_generator=LinearGenerator(start=0.0, step=0.001),
            domain_metadata={
                "logical_size": 4,
                "storage_size": 8,
                "complex_pair_encoding": "separated",
                "first_component": "real",
            },
        ),
    ]
    nmr_data = NMRData(data=data_array, dimensions=dims, source_format="delta")

    result = complexify_indirect_dim(nmr_data, mode="auto")

    assert result.shape == (3, 4)
    np.testing.assert_array_equal(result.data.real, real)
    np.testing.assert_array_equal(result.data.imag, imag)
    assert result.dimensions[1].size == 4
    assert result.dimensions[1].domain_metadata["complexified_mode"] == "separated"
