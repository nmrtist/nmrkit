import numpy as np
import pytest
from nmrkit.core.data import NMRData, DimensionInfo, LinearGenerator
from nmrkit.processing.window import (
    exponential,
    gaussian,
    sine,
    cosine,
    trapezoidal,
    first_point_scaling,
)


# Helper function to create test data with time domain
def create_test_1d_data(
    size=128, complex=True, domain_type="time", can_ft=True, time_step=0.1
):
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
            axis_generator=LinearGenerator(start=0.0, step=time_step),
        )
    ]

    return NMRData(data=data_array, dimensions=dims)


# Test exponential window function
def test_exponential_window_basic():
    # Test basic exponential window application
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")
    original_data = nmr_data.data.copy()

    # Apply exponential window with positive line broadening
    result = exponential(nmr_data, dim=0, lb=1.0)

    # Verify data has been modified
    assert not np.array_equal(result.data, original_data)

    # Check that metadata has been updated
    assert "window_type" in result.dimensions[0].domain_metadata
    assert result.dimensions[0].domain_metadata["window_type"] == "exponential"
    assert result.dimensions[0].domain_metadata["window_lb"] == 1.0


def test_exponential_window_parameters():
    # Test exponential window with different parameters
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")

    # Test with negative line broadening (resolution enhancement)
    result_negative = exponential(nmr_data, dim=0, lb=-0.5)
    assert result_negative.dimensions[0].domain_metadata["window_lb"] == -0.5

    # Test with zero line broadening (should not change data much)
    result_zero = exponential(nmr_data, dim=0, lb=0.0)
    # Window should be all ones when lb=0
    expected_window = np.ones(128)
    window_shape = [128]
    window_reshaped = expected_window.reshape(window_shape)
    expected_data = nmr_data.data * window_reshaped
    np.testing.assert_array_almost_equal(result_zero.data, expected_data)


# Test gaussian window function
def test_gaussian_window_basic():
    # Test basic Gaussian window application
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")
    original_data = nmr_data.data.copy()

    # Apply Gaussian window
    result = gaussian(nmr_data, dim=0, gf=0.1, shift=0.0)

    # Verify data has been modified
    assert not np.array_equal(result.data, original_data)

    # Check that metadata has been updated
    assert "window_type" in result.dimensions[0].domain_metadata
    assert result.dimensions[0].domain_metadata["window_type"] == "gaussian"
    assert result.dimensions[0].domain_metadata["window_gf"] == 0.1
    assert result.dimensions[0].domain_metadata["window_shift"] == 0.0


def test_gaussian_window_parameters():
    # Test Gaussian window with different parameters
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")

    # Test with different gf values
    result_small_gf = gaussian(nmr_data, dim=0, gf=0.05, shift=0.0)
    result_large_gf = gaussian(nmr_data, dim=0, gf=0.2, shift=0.0)

    # Check that different gf values produce different results
    assert not np.array_equal(result_small_gf.data, result_large_gf.data)

    # Test with shift parameter
    result_shifted = gaussian(nmr_data, dim=0, gf=0.1, shift=5.0)
    assert result_shifted.dimensions[0].domain_metadata["window_shift"] == 5.0
    assert not np.array_equal(result_shifted.data, nmr_data.data)


# Test sine window function
def test_sine_window_basic():
    # Test basic sinebell window application
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")
    original_data = nmr_data.data.copy()

    # Apply sinebell window (sb > 0)
    result_sine = sine(nmr_data, dim=0, sb=1.0, shift=0.0)

    # Verify data has been modified
    assert not np.array_equal(result_sine.data, original_data)

    # Check that metadata has been updated
    assert "window_type" in result_sine.dimensions[0].domain_metadata
    assert result_sine.dimensions[0].domain_metadata["window_type"] == "sinebell"
    assert result_sine.dimensions[0].domain_metadata["window_sb"] == 1.0


def test_sine_window_squared():
    # Test squared sinebell window (sb < 0)
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")

    # Apply squared sinebell window
    result_sine2 = sine(nmr_data, dim=0, sb=-1.0, shift=0.0)

    # Check that window type is correctly set
    assert result_sine2.dimensions[0].domain_metadata["window_type"] == "sinebell2"
    assert result_sine2.dimensions[0].domain_metadata["window_sb"] == -1.0


def test_sine_window_parameters():
    # Test sine window with different parameters
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")

    # Test with different sb values
    result_small_sb = sine(nmr_data, dim=0, sb=0.5, shift=0.0)
    result_large_sb = sine(nmr_data, dim=0, sb=2.0, shift=0.0)

    # Check that different sb values produce different results
    assert not np.array_equal(result_small_sb.data, result_large_sb.data)

    # Test with shift parameter
    result_shifted = sine(nmr_data, dim=0, sb=1.0, shift=5.0)
    assert result_shifted.dimensions[0].domain_metadata["window_shift"] == 5.0


# Test cosine window function
def test_cosine_window_basic():
    # Test basic cosine window application
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")
    original_data = nmr_data.data.copy()

    # Apply cosine window
    result = cosine(nmr_data, dim=0, squared=False)

    # Verify data has been modified
    assert not np.array_equal(result.data, original_data)

    # Check that metadata has been updated
    assert "window_type" in result.dimensions[0].domain_metadata
    assert result.dimensions[0].domain_metadata["window_type"] == "cosine"
    assert result.dimensions[0].domain_metadata["window_squared"] == False


def test_cosine_window_squared():
    # Test squared cosine window
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")

    # Apply squared cosine window
    result = cosine(nmr_data, dim=0, squared=True)

    # Check that window type is correctly set
    assert result.dimensions[0].domain_metadata["window_type"] == "cosine2"
    assert result.dimensions[0].domain_metadata["window_squared"]


def test_cosine_window_symmetry():
    # Test that cosine window is symmetric
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")

    # Create test data with ones
    nmr_data.data = np.ones(128, dtype=np.complex128)

    # Apply cosine window
    result = cosine(nmr_data, dim=0, squared=False)

    # Check symmetry
    window = result.data.real  # Since data was ones, window is the real part
    for i in range(128):
        assert np.isclose(window[i], window[127 - i])


# Test trapezoidal window function
def test_trapezoidal_window_basic():
    # Test basic trapezoidal window application
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")
    original_data = nmr_data.data.copy()

    # Apply trapezoidal window (which should be all ones)
    result = trapezoidal(nmr_data, dim=0)

    # Check that metadata has been updated
    assert "window_type" in result.dimensions[0].domain_metadata
    assert result.dimensions[0].domain_metadata["window_type"] == "trapezoidal"

    # Trapezoidal window in this implementation is just ones, so data should
    # be unchanged
    np.testing.assert_array_equal(result.data, original_data)


# Test first_point_scaling function
def test_first_point_scaling_basic():
    # Test basic first point scaling
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")
    original_data = nmr_data.data.copy()

    # Scale first point by 0.5
    result = first_point_scaling(nmr_data, dim=0, factor=0.5)

    # Verify data has been modified only at first point
    assert not np.array_equal(result.data, original_data)

    # Check that first point has been scaled
    assert np.isclose(result.data[0], original_data[0] * 0.5)

    # Check that other points are unchanged
    np.testing.assert_array_equal(result.data[1:], original_data[1:])

    # Check that metadata has been updated
    assert "first_point_scaled" in result.dimensions[0].domain_metadata
    assert result.dimensions[0].domain_metadata["first_point_scaled"]
    assert result.dimensions[0].domain_metadata["first_point_factor"] == 0.5


def test_first_point_scaling_different_factors():
    # Test first point scaling with different factors
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")

    # Test with factor 1.0 (no change)
    result_1 = first_point_scaling(nmr_data, dim=0, factor=1.0)
    np.testing.assert_array_equal(result_1.data, nmr_data.data)

    # Test with factor 2.0 (double first point)
    result_2 = first_point_scaling(nmr_data, dim=0, factor=2.0)
    assert np.isclose(result_2.data[0], nmr_data.data[0] * 2.0)
    np.testing.assert_array_equal(result_2.data[1:], nmr_data.data[1:])


# Test error handling for all window functions
def test_window_functions_invalid_dimension():
    # Test that all window functions raise error for invalid dimension
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")

    # Test with dimension out of range
    with pytest.raises(ValueError):
        exponential(nmr_data, dim=1)

    with pytest.raises(ValueError):
        gaussian(nmr_data, dim=1)

    with pytest.raises(ValueError):
        sine(nmr_data, dim=1)

    with pytest.raises(ValueError):
        cosine(nmr_data, dim=1)

    with pytest.raises(ValueError):
        trapezoidal(nmr_data, dim=1)

    with pytest.raises(ValueError):
        first_point_scaling(nmr_data, dim=1)


# Test window functions with frequency domain data
def test_window_functions_frequency_domain():
    # Test that window functions work with frequency domain data
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="frequency")

    # Apply exponential window (should still work)
    result = exponential(nmr_data, dim=0, lb=1.0)
    assert "window_type" in result.dimensions[0].domain_metadata

    # Apply gaussian window (should still work)
    result = gaussian(nmr_data, dim=0, gf=0.1)
    assert "window_type" in result.dimensions[0].domain_metadata
