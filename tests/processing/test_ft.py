import numpy as np
import pytest
from nmrkit.core.data import NMRData, DimensionInfo, LinearGenerator
from nmrkit.processing.ft import fourier_transform, ft_shift, ft_unshift


# Helper function to create test data
def create_test_1d_data(size=128, complex=True, can_ft=True, domain_type="time"):
    if complex:
        data_array = np.random.rand(size) + 1j * np.random.rand(size)
    else:
        data_array = np.random.rand(size)

    dims = [
        DimensionInfo(
            size=size,
            is_complex=complex,
            domain_type=domain_type,
            can_ft=can_ft,
            axis_generator=LinearGenerator(start=0.0, step=0.1),
        )
    ]

    return NMRData(data=data_array, dimensions=dims)


# Test Fourier transform function
def test_fourier_transform_basic():
    # Test forward FT
    nmr_data = create_test_1d_data(
        size=128, complex=True, can_ft=True, domain_type="time"
    )
    result = fourier_transform(nmr_data, dim=0, inverse=False, shift=True)

    assert result.ndim == 1
    assert result.shape == (128,)
    assert result.is_complex
    assert result.dimensions[0].domain_type == "frequency"
    assert result.dimensions[0].unit == "Hz"

    # Test inverse FT
    result_ifft = fourier_transform(result, dim=0, inverse=True, shift=False)
    assert result_ifft.dimensions[0].domain_type == "time"
    assert result_ifft.dimensions[0].unit == "s"


def test_fourier_transform_parameters():
    # Test with explicit shift parameter
    nmr_data = create_test_1d_data(
        size=128, complex=True, can_ft=True, domain_type="time"
    )

    # Forward FT with no shift
    result_no_shift = fourier_transform(nmr_data, dim=0, inverse=False, shift=False)
    assert "ft_shifted" in result_no_shift.dimensions[0].domain_metadata
    assert result_no_shift.dimensions[0].domain_metadata["ft_shifted"] == False

    # Forward FT with shift
    result_shift = fourier_transform(nmr_data, dim=0, inverse=False, shift=True)
    assert result_shift.dimensions[0].domain_metadata["ft_shifted"]

    # Inverse FT with shift
    result_ifft_shift = fourier_transform(result_shift, dim=0, inverse=True, shift=True)
    assert result_ifft_shift.dimensions[0].domain_metadata["ft_shifted"]


def test_fourier_transform_invalid_parameters():
    # Test with invalid dimension
    nmr_data = create_test_1d_data(
        size=128, complex=True, can_ft=True, domain_type="time"
    )
    with pytest.raises(ValueError):
        fourier_transform(nmr_data, dim=1)  # Dimension out of range

    # Test with non-FT capable dimension
    nmr_data = create_test_1d_data(
        size=128, complex=True, can_ft=False, domain_type="time"
    )
    with pytest.raises(ValueError):
        fourier_transform(nmr_data, dim=0)  # Dimension not FT capable

    # Test with non-complex data
    nmr_data = create_test_1d_data(
        size=128, complex=False, can_ft=True, domain_type="time"
    )
    result = fourier_transform(nmr_data, dim=0)
    assert result.is_complex  # FT should produce complex data


def test_fourier_transform_axis_generator_update():
    # Test that axis generator is properly updated after FT
    nmr_data = create_test_1d_data(
        size=128, complex=True, can_ft=True, domain_type="time"
    )
    nmr_data.dimensions[0].axis_generator = LinearGenerator(start=0.0, step=0.1)

    result = fourier_transform(nmr_data, dim=0, shift=True)

    # Check that axis generator has been updated for frequency domain
    assert isinstance(result.dimensions[0].axis_generator, LinearGenerator)
    # Frequency increment should be 1/(size * time_step)
    expected_step = 1.0 / (128 * 0.1)  # 128 points, 0.1s step
    assert result.dimensions[0].axis_generator.step == expected_step


def test_fourier_transform_indirect_dimension_keeps_full_spectrum():
    data_array = np.random.rand(16, 32) + 1j * np.random.rand(16, 32)
    dims = [
        DimensionInfo(
            size=16,
            is_complex=True,
            domain_type="frequency",
            can_ft=False,
        ),
        DimensionInfo(
            size=32,
            is_complex=True,
            domain_type="time",
            can_ft=True,
            axis_generator=LinearGenerator(start=0.0, step=0.001),
        ),
    ]
    nmr_data = NMRData(data=data_array, dimensions=dims)

    result = fourier_transform(nmr_data, dim=1, shift=True)

    assert result.shape == (16, 32)
    assert result.dimensions[1].size == 32
    assert result.dimensions[1].domain_type == "frequency"


def test_fourier_transform_uses_metadata_fft_sign():
    size = 16
    data_array = np.random.rand(size) + 1j * np.random.rand(size)
    dims = [
        DimensionInfo(
            size=size,
            is_complex=True,
            domain_type="time",
            can_ft=True,
            axis_generator=LinearGenerator(start=0.0, step=0.001),
            domain_metadata={"fft_sign": -1},
        )
    ]
    nmr_data = NMRData(data=data_array, dimensions=dims)

    result = fourier_transform(nmr_data, dim=0, shift=True)
    expected = np.fft.fftshift(np.fft.ifftn(data_array, axes=(0,)) * size)

    np.testing.assert_allclose(result.data, expected)
    assert result.dimensions[0].domain_metadata["fft_sign"] == -1


def test_ft_shift_functions():
    # Test ft_shift and ft_unshift functions
    nmr_data = create_test_1d_data(
        size=128, complex=True, can_ft=True, domain_type="time"
    )

    # Test basic shift
    result = ft_shift(nmr_data, dim=0, shift=True)
    assert "ft_shifted" in result.dimensions[0].domain_metadata
    assert result.dimensions[0].domain_metadata["ft_shifted"]

    # Test unshift (ft_shift with shift=False)
    result_unshifted = ft_shift(result, dim=0, shift=False)
    assert result_unshifted.dimensions[0].domain_metadata["ft_shifted"] == False

    # Test ft_unshift function (should be equivalent to ft_shift with
    # shift=False)
    result_unshifted2 = ft_unshift(result, dim=0)
    assert result_unshifted2.dimensions[0].domain_metadata["ft_shifted"] == False

    # Test with invalid dimension
    with pytest.raises(ValueError):
        ft_shift(nmr_data, dim=1)  # Dimension out of range


def test_fourier_transform_consistency():
    # Test that FT followed by IFT returns the original data (up to numerical
    # precision)
    nmr_data = create_test_1d_data(
        size=128, complex=True, can_ft=True, domain_type="time"
    )
    original_data = nmr_data.data.copy()

    # Test with different shift combinations
    # Case 1: No shift for either
    result_ft1 = fourier_transform(nmr_data, dim=0, shift=False)
    result_ift1 = fourier_transform(result_ft1, dim=0, inverse=True, shift=False)

    # Account for scaling factor: inverse FT scales by size, forward FT scales by 1
    # So we need to normalize by the size to get back the original data
    size = nmr_data.dimensions[0].size

    # Check that the magnitude is approximately the same (phase differences
    # are expected)
    np.testing.assert_array_almost_equal(
        np.abs(original_data), np.abs(result_ift1.data * size), decimal=4
    )

    # Case 2: Shift for forward FT, no shift for inverse FT
    result_ft2 = fourier_transform(nmr_data, dim=0, shift=True)
    result_ift2 = fourier_transform(result_ft2, dim=0, inverse=True, shift=False)

    # Check that the magnitude is approximately the same (accounting for
    # scaling)
    np.testing.assert_array_almost_equal(
        np.abs(original_data), np.abs(result_ift2.data * size), decimal=4
    )
