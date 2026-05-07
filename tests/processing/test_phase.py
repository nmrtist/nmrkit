import numpy as np
import pytest
from nmrkit.core.data import NMRData, DimensionInfo, LinearGenerator
from nmrkit.processing.phase import (
    phase_correct,
    correct_digital_filter_phase,
    compensate_digital_filter_delay,
    autophase,
)


# Helper function to create test data
def create_test_1d_data(size=128, complex=True, domain_type="frequency", can_ft=True):
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


# Test phase_correct function
def test_phase_correct_basic():
    # Test basic phase correction functionality
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="frequency")
    original_data = nmr_data.data.copy()

    # Apply zero-order phase correction (90 degrees)
    result = phase_correct(nmr_data, dim=0, ph0=90.0, ph1=0.0)

    # Verify data has been modified
    assert not np.array_equal(result.data, original_data)

    # Check that metadata has been updated
    assert "phase_correction" in result.dimensions[0].domain_metadata
    correction = result.dimensions[0].domain_metadata["phase_correction"][-1]
    assert correction["ph0"] == 90.0
    assert correction["ph1"] == 0.0
    assert correction["pivot"] == 64  # Default pivot at center


def test_phase_correct_zero_order():
    # Test zero-order phase correction
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="frequency")

    # Create test data with known phase
    data_array = np.ones(128, dtype=np.complex128)
    nmr_data.data = data_array

    # Apply 0 degrees phase correction (should not change data)
    result_0 = phase_correct(nmr_data, dim=0, ph0=0.0)
    np.testing.assert_array_equal(result_0.data, data_array)

    # Apply 180 degrees phase correction (should invert sign)
    result_180 = phase_correct(nmr_data, dim=0, ph0=180.0)
    expected_180 = np.ones(128, dtype=np.complex128) * np.exp(1j * np.pi)
    np.testing.assert_array_almost_equal(result_180.data, expected_180)


def test_phase_correct_first_order():
    # Test first-order phase correction
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="frequency")

    # Create test data
    data_array = np.ones(128, dtype=np.complex128)
    nmr_data.data = data_array

    # Apply first-order phase correction
    result = phase_correct(nmr_data, dim=0, ph0=0.0, ph1=360.0, pivot=64)

    # Check that phase changes linearly across the spectrum
    angles = np.angle(result.data)

    # Calculate expected angles using the same formula as in the code
    indices = np.arange(128)
    expected_angles = np.deg2rad(360.0) * (indices - 64) / (128 - 1)

    # Normalize both actual and expected angles to the same range
    angles_normalized = np.angle(np.exp(1j * angles))
    expected_normalized = np.angle(np.exp(1j * expected_angles))

    np.testing.assert_array_almost_equal(
        angles_normalized, expected_normalized, decimal=4
    )


def test_phase_correct_pivot():
    # Test phase correction with different pivot points
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="frequency")

    # Create test data
    data_array = np.ones(128, dtype=np.complex128)
    nmr_data.data = data_array

    # Apply first-order phase correction with pivot at beginning
    result_start = phase_correct(nmr_data, dim=0, ph0=0.0, ph1=360.0, pivot=0)
    angles_start = np.angle(result_start.data)

    # Calculate expected angles using the same formula as in the code
    indices = np.arange(128)
    expected_angles_start = np.deg2rad(360.0) * (indices - 0) / (128 - 1)

    # Normalize both actual and expected angles to the same range
    angles_start_normalized = np.angle(np.exp(1j * angles_start))
    expected_start_normalized = np.angle(np.exp(1j * expected_angles_start))

    np.testing.assert_array_almost_equal(
        angles_start_normalized, expected_start_normalized, decimal=4
    )

    # Apply first-order phase correction with pivot at end
    result_end = phase_correct(nmr_data, dim=0, ph0=0.0, ph1=360.0, pivot=127)
    angles_end = np.angle(result_end.data)

    # Calculate expected angles using the same formula as in the code
    expected_angles_end = np.deg2rad(360.0) * (indices - 127) / (128 - 1)

    # Normalize both actual and expected angles to the same range
    angles_end_normalized = np.angle(np.exp(1j * angles_end))
    expected_end_normalized = np.angle(np.exp(1j * expected_angles_end))

    np.testing.assert_array_almost_equal(
        angles_end_normalized, expected_end_normalized, decimal=4
    )


def test_phase_correct_error_handling():
    # Test error handling in phase_correct
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="frequency")

    # Test with invalid dimension
    with pytest.raises(ValueError):
        phase_correct(nmr_data, dim=1)  # Dimension out of range

    # Test with invalid pivot point
    with pytest.raises(ValueError):
        phase_correct(nmr_data, dim=0, pivot=-1)  # Pivot too low

    with pytest.raises(ValueError):
        phase_correct(nmr_data, dim=0, pivot=128)  # Pivot too high


# Test correct_digital_filter_phase function
def test_correct_digital_filter_phase_basic():
    # Test basic digital filter phase correction
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="frequency")
    original_data = nmr_data.data.copy()

    # Apply correction with group delay
    result = correct_digital_filter_phase(nmr_data, dim=0, group_delay=10.0)

    # Verify data has been modified
    assert not np.array_equal(result.data, original_data)

    # Check that metadata has been updated
    assert "phase_correction" in result.dimensions[0].domain_metadata
    correction = result.dimensions[0].domain_metadata["phase_correction"][-1]
    assert correction["type"] == "digital_filter"
    assert correction["group_delay"] == 10.0


def test_correct_digital_filter_phase_metadata():
    # Test group delay extraction from metadata
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="frequency")

    # Add TopSpin-style metadata with group delay
    nmr_data.source_format = "topspin"
    nmr_data.metadata = {"parameters": {"direct": {"GRPDLY": 5.0}}}

    # Apply correction without specifying group delay
    result = correct_digital_filter_phase(nmr_data, dim=0)

    # Check that group delay was extracted from metadata
    correction = result.dimensions[0].domain_metadata["phase_correction"][-1]
    assert correction["group_delay"] == 5.0


def test_correct_digital_filter_phase_indirect_dimension():
    # Test correction for indirect dimensions
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="frequency")

    # Add TopSpin-style metadata with group delay for indirect dimension
    nmr_data.source_format = "topspin"
    nmr_data.metadata = {"parameters": {"indirect1": {"GRPDLY": 7.5}}}

    # Apply correction (should use default since dim 0 doesn't have indirect1
    # metadata)
    result = correct_digital_filter_phase(nmr_data, dim=0)
    correction = result.dimensions[0].domain_metadata["phase_correction"][-1]
    assert correction["group_delay"] == 0.0  # Default value


def test_correct_digital_filter_phase_error_handling():
    # Test error handling in correct_digital_filter_phase
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="frequency")

    # Test with invalid dimension
    with pytest.raises(ValueError):
        correct_digital_filter_phase(nmr_data, dim=1)  # Dimension out of range


# Test autophase function
def create_absorptive_test_spectrum(size=512):
    """Create a simple 1D absorptive spectrum with positive Lorentzian peaks."""
    x = np.linspace(-1.0, 1.0, size)
    real = (
        1.0 / (1.0 + ((x + 0.35) / 0.03) ** 2)
        + 0.7 / (1.0 + ((x - 0.10) / 0.05) ** 2)
        + 0.5 / (1.0 + ((x - 0.42) / 0.02) ** 2)
    )
    imag = np.zeros_like(real)
    return real + 1j * imag


def test_autophase_global_recovers_known_phase():
    nmr_data = create_test_1d_data(size=512, complex=True, domain_type="frequency")
    reference = create_absorptive_test_spectrum(size=512)
    nmr_data.data = reference.copy()

    phased = phase_correct(nmr_data, dim=0, ph0=52.0, ph1=-110.0, pivot=256)
    result = autophase(phased, dim=0, pivot=256, maxiter=300)

    np.testing.assert_allclose(
        np.real(result.data),
        np.real(reference),
        atol=3e-2,
        rtol=5e-2,
    )
    assert np.linalg.norm(np.imag(result.data)) < 0.15 * np.linalg.norm(
        np.real(reference)
    )

    correction = result.dimensions[0].domain_metadata["phase_correction"][-1]
    assert abs(correction["pivot"] - 256) == 0


def test_autophase_zero_order_only():
    nmr_data = create_test_1d_data(size=512, complex=True, domain_type="frequency")
    reference = create_absorptive_test_spectrum(size=512)
    nmr_data.data = reference.copy()

    phased = phase_correct(nmr_data, dim=0, ph0=87.0, ph1=0.0, pivot=256)
    result = autophase(phased, dim=0, method="zero_order", pivot=256)

    np.testing.assert_allclose(
        np.real(result.data),
        np.real(reference),
        atol=2e-2,
        rtol=5e-2,
    )
    correction = result.dimensions[0].domain_metadata["phase_correction"][-1]
    assert correction["ph1"] == 0.0


def test_autophase_invalid_method():
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="frequency")

    with pytest.raises(ValueError):
        autophase(nmr_data, dim=0, method="peak_minima")


# Tests for JEOL Delta digital filter group delay


def test_correct_digital_filter_phase_delta_metadata():
    """Test group delay extraction from Delta metadata."""
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="frequency")

    nmr_data.source_format = "delta"
    nmr_data.metadata = {"digital_filter_group_delay": 19.5}

    result = correct_digital_filter_phase(nmr_data, dim=0)
    correction = result.dimensions[0].domain_metadata["phase_correction"][-1]
    assert correction["group_delay"] == 19.5


def test_compensate_digital_filter_delay_basic():
    """Test time-domain circular shift removes pre-echo."""
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")
    original = nmr_data.data.copy()

    # Simulate pre-echo: first 10 points are near-zero, rest is signal
    nmr_data.data[:10] = 0.001 * (np.random.rand(10) + 1j * np.random.rand(10))
    nmr_data.data[10:] = 10.0 * (np.random.rand(118) + 1j * np.random.rand(118))

    result = compensate_digital_filter_delay(nmr_data, group_delay=10.0)

    # After shift, first point should be what was at index 10 (large signal)
    assert np.abs(result.data[0]) > 1.0
    # Last 10 points should be zeroed
    np.testing.assert_array_equal(result.data[-10:], 0)
    # Metadata should record the correction
    correction = result.dimensions[0].domain_metadata["phase_correction"][-1]
    assert correction["type"] == "digital_filter_removal"
    assert correction["shift"] == 10


def test_compensate_digital_filter_delay_auto_metadata():
    """Test auto group delay extraction from Delta metadata."""
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")
    nmr_data.source_format = "delta"
    nmr_data.metadata = {"digital_filter_group_delay": 5.7}

    result = compensate_digital_filter_delay(nmr_data, dim=0)
    correction = result.dimensions[0].domain_metadata["phase_correction"][-1]
    # 5.7 rounds to 6
    assert correction["shift"] == 6


def test_compensate_digital_filter_delay_zero_delay():
    """Test that zero group delay returns unchanged copy."""
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")
    original = nmr_data.data.copy()

    result = compensate_digital_filter_delay(nmr_data, group_delay=0.0)
    np.testing.assert_array_equal(result.data, original)
