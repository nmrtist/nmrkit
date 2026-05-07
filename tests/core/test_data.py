import numpy as np
import pytest
from nmrkit.core.data import (
    AxisGenerator,
    LinearGenerator,
    ExponentialGenerator,
    NonUniformGenerator,
    DimensionInfo,
    NMRData,
)


# Test AxisGenerator related classes
def test_linear_generator():
    # Test basic functionality
    generator = LinearGenerator(start=0.0, step=0.1)
    axis = generator.generate(5)
    expected = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    np.testing.assert_array_almost_equal(axis, expected, decimal=15)
    assert generator.is_uniform

    # Test different parameters
    generator = LinearGenerator(start=1.0, step=0.5)
    axis = generator.generate(3)
    expected = np.array([1.0, 1.5, 2.0])
    np.testing.assert_array_almost_equal(axis, expected, decimal=15)

    # Test edge cases
    generator = LinearGenerator(start=0.0, step=0.0)
    axis = generator.generate(5)
    expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(axis, expected)


def test_exponential_generator():
    # Test basic functionality
    generator = ExponentialGenerator(start=0.0, growth_rate=1.1)
    axis = generator.generate(3)
    expected = np.array([0.0, 0.1, 0.21])
    np.testing.assert_array_almost_equal(axis, expected, decimal=4)
    assert not generator.is_uniform

    # Test different parameters
    generator = ExponentialGenerator(start=1.0, growth_rate=2.0)
    axis = generator.generate(3)
    expected = np.array([1.0, 2.0, 4.0])
    np.testing.assert_array_equal(axis, expected)


def test_nonuniform_generator():
    # Test basic functionality
    values = np.array([0.1, 0.3, 0.7, 1.5])
    generator = NonUniformGenerator(values)
    axis = generator.generate(4)
    np.testing.assert_array_equal(axis, values)
    assert not generator.is_uniform

    # Test error case
    with pytest.raises(ValueError):
        generator.generate(3)  # Size mismatch


# Test DimensionInfo class
def test_dimension_info():
    # Test basic initialization
    dim = DimensionInfo(
        size=1024,
        is_complex=True,
        spectral_width=12000.0,
        observation_frequency=600.0,
        nucleus="1H",
        domain_type="time",
        can_ft=True,
        unit="s",
    )

    assert dim.size == 1024
    assert dim.is_complex
    assert dim.spectral_width == 12000.0
    assert dim.observation_frequency == 600.0
    assert dim.nucleus == "1H"
    assert dim.domain_type == "time"
    assert dim.can_ft
    assert dim.unit == "s"

    # Test axis generation
    axis = dim.generate_axis()
    assert len(axis) == 1024

    # Test default values
    dim = DimensionInfo(size=512)
    assert dim.size == 512
    assert not dim.is_complex
    assert dim.domain_type is None
    assert not dim.can_ft
    assert dim.unit is None

    # Test error case
    with pytest.raises(ValueError):
        DimensionInfo(size=0)  # Zero size


def test_dimension_info_post_init():
    # Test default axis generator
    dim = DimensionInfo(size=10)
    assert isinstance(dim.axis_generator, LinearGenerator)

    # Test automatic unit setting
    dim = DimensionInfo(size=10, domain_type="time")
    assert dim.unit == "s"

    dim = DimensionInfo(size=10, domain_type="frequency")
    assert dim.unit is None  # Frequency domain unit not automatically set


# Test NMRData class
def test_nmr_data_init():
    # Test basic initialization
    data_array = np.random.rand(1024) + 1j * np.random.rand(1024)
    dims = [DimensionInfo(size=1024, is_complex=True, domain_type="time", can_ft=True)]
    nmr_data = NMRData(data=data_array, dimensions=dims)

    assert nmr_data.ndim == 1
    assert nmr_data.shape == (1024,)
    assert nmr_data.dtype == np.complex128
    assert nmr_data.is_complex

    # Test metadata
    metadata = {"experiment": "1D 1H"}
    nmr_data = NMRData(data=data_array, dimensions=dims, metadata=metadata)
    assert nmr_data.metadata["experiment"] == "1D 1H"

    # Test error cases
    with pytest.raises(ValueError):
        # Dimension count mismatch
        NMRData(data=np.random.rand(1024, 512), dimensions=dims)

    with pytest.raises(ValueError):
        # Dimension size mismatch
        dims[0].size = 512
        NMRData(data=data_array, dimensions=dims)


def test_nmr_data_copy():
    # Test copy functionality
    data_array = np.random.rand(1024)
    dims = [DimensionInfo(size=1024)]
    nmr_data = NMRData(data=data_array, dimensions=dims)

    copy_data = nmr_data.copy()
    assert copy_data is not nmr_data
    assert np.array_equal(copy_data.data, nmr_data.data)
    assert copy_data.dimensions is not nmr_data.dimensions

    # Test that modification doesn't affect original
    copy_data.data[0] = 100.0
    assert nmr_data.data[0] != 100.0


def test_nmr_data_properties():
    # Test multi-dimensional data
    data_array = np.random.rand(256, 128)
    dims = [
        DimensionInfo(size=256, domain_type="time"),
        DimensionInfo(size=128, domain_type="time"),
    ]
    nmr_data = NMRData(data=data_array, dimensions=dims)

    assert nmr_data.ndim == 2
    assert nmr_data.shape == (256, 128)
    assert nmr_data.full_shape == (256, 128)
