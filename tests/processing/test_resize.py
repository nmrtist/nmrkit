import numpy as np
import pytest
from nmrkit.core.data import NMRData, DimensionInfo, LinearGenerator
from nmrkit.processing.resize import zero_fill, extract_region


# Helper function to create test data
def create_test_1d_data(size=128, complex=True, domain_type="time", can_ft=True):
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


def create_test_2d_data(
    size1=128, size2=64, complex=True, domain_type="time", can_ft=True
):
    if complex:
        data_array = np.random.rand(size1, size2) + 1j * np.random.rand(size1, size2)
    else:
        data_array = np.random.rand(size1, size2)

    dims = [
        DimensionInfo(
            size=size1,
            is_complex=complex,
            domain_type=domain_type,
            can_ft=can_ft,
            axis_generator=LinearGenerator(start=0.0, step=0.1),
        ),
        DimensionInfo(
            size=size2,
            is_complex=complex,
            domain_type=domain_type,
            can_ft=can_ft,
            axis_generator=LinearGenerator(start=0.0, step=0.2),
        ),
    ]

    return NMRData(data=data_array, dimensions=dims)


# Test zero_fill function
def test_zero_fill_basic():
    # Test basic zero filling functionality
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")
    original_data = nmr_data.data.copy()

    # Apply zero filling to next power of two
    result = zero_fill(nmr_data, dim=0, size=256, power_of_two=False)

    # Verify data has been extended
    assert result.shape[0] == 256

    # Check that original data is preserved and zeros are added at the end
    np.testing.assert_array_equal(result.data[:128], original_data)
    np.testing.assert_array_equal(result.data[128:], np.zeros(128, dtype=np.complex128))

    # Check that dimension information has been updated
    assert result.dimensions[0].size == 256

    # Check that metadata has been updated
    assert result.dimensions[0].domain_metadata["zero_filled"]
    assert result.dimensions[0].domain_metadata["original_size"] == 128
    assert result.dimensions[0].domain_metadata["target_size"] == 256


def test_zero_fill_power_of_two():
    # Test zero filling to next power of two
    nmr_data = create_test_1d_data(size=100, complex=True, domain_type="time")

    # Apply zero filling with power_of_two=True (size=None)
    result = zero_fill(nmr_data, dim=0, size=None, power_of_two=True)

    # Check that size has been rounded up to next power of two (128)
    assert result.dimensions[0].size == 128
    assert result.data.shape[0] == 128

    # Verify original data is preserved
    np.testing.assert_array_equal(result.data[:100], nmr_data.data)


def test_zero_fill_no_change():
    # Test zero filling with same size (should return copy)
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")

    # Apply zero filling with same size
    result = zero_fill(nmr_data, dim=0, size=128, power_of_two=False)

    # Verify data is the same but object is different
    assert result is not nmr_data
    np.testing.assert_array_equal(result.data, nmr_data.data)


def test_zero_fill_2d():
    # Test zero filling on 2D data
    nmr_data = create_test_2d_data(
        size1=128, size2=64, complex=True, domain_type="time"
    )
    original_data = nmr_data.data.copy()

    # Apply zero filling to second dimension
    result = zero_fill(nmr_data, dim=1, size=128, power_of_two=False)

    # Check that only the specified dimension has been extended
    assert result.shape == (128, 128)
    assert result.dimensions[0].size == 128  # First dimension unchanged
    assert result.dimensions[1].size == 128  # Second dimension extended

    # Verify original data is preserved
    np.testing.assert_array_equal(result.data[:, :64], original_data)
    np.testing.assert_array_equal(
        result.data[:, 64:], np.zeros((128, 64), dtype=np.complex128)
    )


def test_zero_fill_error_handling():
    # Test error handling in zero_fill
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")

    # Test with invalid dimension
    with pytest.raises(ValueError):
        zero_fill(nmr_data, dim=1)  # Dimension out of range

    # Test with target size smaller than current size
    with pytest.raises(ValueError):
        zero_fill(nmr_data, dim=0, size=64, power_of_two=False)

    # Test with negative dimension
    with pytest.raises(ValueError):
        zero_fill(nmr_data, dim=-1)


def test_zero_fill_no_size_power_of_two_false():
    # Test zero_fill with size=None and power_of_two=False
    nmr_data = create_test_1d_data(size=100, complex=True, domain_type="time")

    # Should return copy without changes
    result = zero_fill(nmr_data, dim=0, size=None, power_of_two=False)

    assert result is not nmr_data
    assert result.dimensions[0].size == 100
    np.testing.assert_array_equal(result.data, nmr_data.data)


# Test extract_region function
def test_extract_region_basic():
    # Test basic region extraction
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")
    original_data = nmr_data.data.copy()

    # Extract a region from the middle
    result = extract_region(nmr_data, dim=0, start=32, end=96)

    # Verify data has been extracted
    assert result.shape[0] == 64  # 96 - 32 = 64
    assert result.dimensions[0].size == 64

    # Check that extracted region matches original data
    np.testing.assert_array_equal(result.data, original_data[32:96])

    # Check that metadata has been updated
    assert result.dimensions[0].domain_metadata["extracted"]
    assert result.dimensions[0].domain_metadata["extraction_start"] == 32
    assert result.dimensions[0].domain_metadata["extraction_end"] == 96


def test_extract_region_start_only():
    # Test extraction from start to end of data
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")

    # Extract from start=64 to end (None)
    result = extract_region(nmr_data, dim=0, start=64, end=None)

    # Verify data has been extracted correctly
    assert result.shape[0] == 64  # 128 - 64 = 64
    np.testing.assert_array_equal(result.data, nmr_data.data[64:])


def test_extract_region_end_of_data():
    # Test extraction from beginning to middle
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")

    # Extract first 64 points
    result = extract_region(nmr_data, dim=0, start=0, end=64)

    # Verify data has been extracted correctly
    assert result.shape[0] == 64
    np.testing.assert_array_equal(result.data, nmr_data.data[:64])


def test_extract_region_2d():
    # Test region extraction on 2D data
    nmr_data = create_test_2d_data(
        size1=128, size2=64, complex=True, domain_type="time"
    )
    original_data = nmr_data.data.copy()

    # Extract region from second dimension
    result = extract_region(nmr_data, dim=1, start=16, end=48)

    # Check that only the specified dimension has been extracted
    assert result.shape == (128, 32)  # 48 - 16 = 32
    assert result.dimensions[0].size == 128  # First dimension unchanged
    assert result.dimensions[1].size == 32  # Second dimension extracted

    # Verify extracted region matches original data
    np.testing.assert_array_equal(result.data, original_data[:, 16:48])


def test_extract_region_error_handling():
    # Test error handling in extract_region
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")

    # Test with invalid dimension
    with pytest.raises(ValueError):
        extract_region(nmr_data, dim=1)  # Dimension out of range

    # Test with negative start index
    with pytest.raises(ValueError):
        extract_region(nmr_data, dim=0, start=-1, end=64)

    # Test with end index larger than current size
    with pytest.raises(ValueError):
        extract_region(nmr_data, dim=0, start=0, end=256)

    # Test with start >= end
    with pytest.raises(ValueError):
        extract_region(nmr_data, dim=0, start=64, end=32)


def test_extract_region_entire_data():
    # Test extracting the entire data (should return copy)
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")

    # Extract entire data
    result = extract_region(nmr_data, dim=0, start=0, end=128)

    # Verify data is the same but object is different
    assert result is not nmr_data
    np.testing.assert_array_equal(result.data, nmr_data.data)


# Test integration of zero_fill and extract_region
def test_zero_fill_and_extract_region():
    # Test combining zero_fill and extract_region
    nmr_data = create_test_1d_data(size=100, complex=True, domain_type="time")

    # First zero fill to 200 points
    result_zero_fill = zero_fill(nmr_data, dim=0, size=200, power_of_two=False)
    assert result_zero_fill.dimensions[0].size == 200

    # Then extract a region from the zero-filled data
    result_extract = extract_region(result_zero_fill, dim=0, start=50, end=150)
    assert result_extract.dimensions[0].size == 100

    # Verify the extracted region includes both original data and zeros
    # Original data was 100 points, zero filled to 200
    # Extraction from 50-150 includes original data from 50-100 and zeros from
    # 100-150
    np.testing.assert_array_equal(result_extract.data[:50], nmr_data.data[50:100])
    np.testing.assert_array_equal(
        result_extract.data[50:], np.zeros(50, dtype=np.complex128)
    )
