import numpy as np
import pytest
from nmrkit.core.data import NMRData, DimensionInfo, LinearGenerator
from nmrkit.utils import (
    validate_dimension,
    create_dimension_shape,
    update_dimension_info,
    update_domain_metadata,
    get_time_array,
    validate_param_value,
    validate_param_type,
    validate_param_options,
)


# Helper functions to create test data
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


# Test validate_dimension function
def test_validate_dimension_valid():
    # Test valid dimension indices
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")

    # Should not raise an exception
    validate_dimension(nmr_data, 0)

    # Test with 2D data
    nmr_data_2d = create_test_2d_data(
        size1=128, size2=64, complex=True, domain_type="time"
    )
    validate_dimension(nmr_data_2d, 0)
    validate_dimension(nmr_data_2d, 1)


def test_validate_dimension_invalid():
    # Test invalid dimension indices
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")

    # Test negative dimension
    with pytest.raises(ValueError):
        validate_dimension(nmr_data, -1)

    # Test dimension too large
    with pytest.raises(ValueError):
        validate_dimension(nmr_data, 1)

    # Test with 2D data
    nmr_data_2d = create_test_2d_data(
        size1=128, size2=64, complex=True, domain_type="time"
    )
    with pytest.raises(ValueError):
        validate_dimension(nmr_data_2d, 2)


# Test create_dimension_shape function
def test_create_dimension_shape_1d():
    # Test shape creation for 1D data
    shape = create_dimension_shape(ndim=1, dim=0, dim_size=128)
    assert shape == [128]


def test_create_dimension_shape_2d():
    # Test shape creation for 2D data
    # For first dimension
    shape_dim0 = create_dimension_shape(ndim=2, dim=0, dim_size=128)
    assert shape_dim0 == [128, 1]

    # For second dimension
    shape_dim1 = create_dimension_shape(ndim=2, dim=1, dim_size=64)
    assert shape_dim1 == [1, 64]


def test_create_dimension_shape_3d():
    # Test shape creation for 3D data
    # For middle dimension
    shape = create_dimension_shape(ndim=3, dim=1, dim_size=256)
    assert shape == [1, 256, 1]

    # For last dimension
    shape = create_dimension_shape(ndim=3, dim=2, dim_size=512)
    assert shape == [1, 1, 512]


# Test update_dimension_info function
def test_update_dimension_info_basic():
    # Test updating basic DimensionInfo attributes
    dim_info = DimensionInfo(
        size=128,
        is_complex=True,
        domain_type="time",
        can_ft=True,
        axis_generator=LinearGenerator(start=0.0, step=0.1),
    )

    # Update size and domain_type
    updated_dim = update_dimension_info(dim_info, size=256, domain_type="frequency")

    # Verify updates were applied
    assert updated_dim.size == 256
    assert updated_dim.domain_type == "frequency"

    # Verify original attributes are preserved
    assert updated_dim.is_complex
    assert updated_dim.can_ft
    assert updated_dim.axis_generator.step == 0.1


def test_update_dimension_info_all_attributes():
    # Test updating all DimensionInfo attributes
    dim_info = DimensionInfo(
        size=128,
        is_complex=True,
        domain_type="time",
        can_ft=True,
        axis_generator=LinearGenerator(start=0.0, step=0.1),
        nucleus="1H",
        spectral_width=12000.0,
        observation_frequency=600.0,
        unit="s",
    )

    # Create new axis generator
    new_axis_gen = LinearGenerator(start=0.0, step=0.2)

    # Update all attributes
    updated_dim = update_dimension_info(
        dim_info,
        size=512,
        is_complex=False,
        domain_type="frequency",
        can_ft=True,
        axis_generator=new_axis_gen,
        nucleus="13C",
        spectral_width=24000.0,
        observation_frequency=150.0,
        unit="Hz",
    )

    # Verify all updates were applied
    assert updated_dim.size == 512
    assert updated_dim.is_complex == False
    assert updated_dim.domain_type == "frequency"
    assert updated_dim.can_ft
    assert updated_dim.axis_generator.step == 0.2
    assert updated_dim.nucleus == "13C"
    assert updated_dim.spectral_width == 24000.0
    assert updated_dim.observation_frequency == 150.0
    assert updated_dim.unit == "Hz"


# Test update_domain_metadata function
def test_update_domain_metadata_basic():
    # Test updating domain metadata
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")

    # Update metadata for the first dimension
    update_domain_metadata(
        nmr_data, dim=0, window_applied=True, window_type="exponential"
    )

    # Verify metadata was updated
    assert nmr_data.dimensions[0].domain_metadata["window_applied"]
    assert nmr_data.dimensions[0].domain_metadata["window_type"] == "exponential"


def test_update_domain_metadata_multiple_updates():
    # Test multiple metadata updates
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")

    # First update
    update_domain_metadata(nmr_data, dim=0, processing_step="phase_correction")

    # Second update - add more metadata
    update_domain_metadata(nmr_data, dim=0, ph0=0.0, ph1=10.0)

    # Verify all metadata is present
    assert (
        nmr_data.dimensions[0].domain_metadata["processing_step"] == "phase_correction"
    )
    assert nmr_data.dimensions[0].domain_metadata["ph0"] == 0.0
    assert nmr_data.dimensions[0].domain_metadata["ph1"] == 10.0

    # Update existing metadata
    update_domain_metadata(nmr_data, dim=0, ph0=45.0)
    assert nmr_data.dimensions[0].domain_metadata["ph0"] == 45.0


# Test get_time_array function
def test_get_time_array_basic():
    # Test getting time array from DimensionInfo
    dim_info = DimensionInfo(
        size=10,
        is_complex=True,
        domain_type="time",
        can_ft=True,
        axis_generator=LinearGenerator(start=0.0, step=0.1),
    )

    time_array = get_time_array(dim_info)

    # Verify time array is correct
    expected = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    np.testing.assert_array_almost_equal(time_array, expected)


def test_get_time_array_from_nmr_data():
    # Test getting time array from NMRData dimension
    nmr_data = create_test_1d_data(
        size=5, complex=True, domain_type="time", can_ft=True
    )

    time_array = get_time_array(nmr_data.dimensions[0])

    # Verify time array is correct
    expected = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    np.testing.assert_array_almost_equal(time_array, expected)


# Test validate_param_value function
def test_validate_param_value_valid():
    # Test valid parameter values
    # Integer values
    validate_param_value("size", 128, min_value=1, max_value=256)

    # Float values
    validate_param_value("line_broadening", 1.0, min_value=0.0, max_value=10.0)

    # Just minimum value
    validate_param_value("shift", 0.5, min_value=0.0)

    # Just maximum value
    validate_param_value("threshold", 0.8, max_value=1.0)


def test_validate_param_value_invalid():
    # Test invalid parameter values
    # Below minimum
    with pytest.raises(ValueError):
        validate_param_value("size", 0, min_value=1, max_value=256)

    # Above maximum
    with pytest.raises(ValueError):
        validate_param_value("size", 512, min_value=1, max_value=256)

    # Exactly minimum (should pass)
    validate_param_value("size", 1, min_value=1, max_value=256)

    # Exactly maximum (should pass)
    validate_param_value("size", 256, min_value=1, max_value=256)


# Test validate_param_type function
def test_validate_param_type_valid():
    # Test valid parameter types
    # Integer
    validate_param_type("size", 128, (int,))

    # Float
    validate_param_type("threshold", 0.5, (float,))

    # Either int or float
    validate_param_type("size", 128, (int, float))
    validate_param_type("size", 128.5, (int, float))

    # String
    validate_param_type("method", "exponential", (str,))


def test_validate_param_type_invalid():
    # Test invalid parameter types
    # Wrong type
    with pytest.raises(TypeError):
        validate_param_type("size", "128", (int,))

    # Wrong type in union
    with pytest.raises(TypeError):
        validate_param_type("size", [128], (int, float))

    # Boolean instead of int/float
    with pytest.raises(TypeError):
        validate_param_type("size", True, (int, float))


# Test validate_param_options function
def test_validate_param_options_valid():
    # Test valid parameter options
    # String options
    validate_param_options(
        "method", "exponential", ["exponential", "gaussian", "sinebell"]
    )

    # Integer options
    validate_param_options("mode", 1, [0, 1, 2, 3])

    # Boolean options
    validate_param_options("enabled", True, [True, False])


def test_validate_param_options_invalid():
    # Test invalid parameter options
    # Not in options list
    with pytest.raises(ValueError):
        validate_param_options(
            "method", "cosine", ["exponential", "gaussian", "sinebell"]
        )

    # Wrong type in options list
    with pytest.raises(ValueError):
        validate_param_options("mode", 5, [0, 1, 2, 3])

    # Empty string not in options
    with pytest.raises(ValueError):
        validate_param_options("method", "", ["exponential", "gaussian"])


# Test integration of utility functions
def test_utility_functions_integration():
    # Test integration of multiple utility functions
    nmr_data = create_test_1d_data(size=128, complex=True, domain_type="time")

    # Validate dimension (should pass)
    validate_dimension(nmr_data, 0)

    # Update dimension info
    updated_dim = update_dimension_info(
        nmr_data.dimensions[0], domain_type="frequency", unit="Hz"
    )

    # Update domain metadata
    update_domain_metadata(nmr_data, 0, processing_step="ft", ft_applied=True)

    # Validate parameter values
    validate_param_value("size", updated_dim.size, min_value=1)
    validate_param_type("domain_type", updated_dim.domain_type, (str,))
    validate_param_options("unit", updated_dim.unit, ["s", "Hz", "ppm"])

    # Get time array from original dimension
    time_array = get_time_array(nmr_data.dimensions[0])
    assert len(time_array) == 128
