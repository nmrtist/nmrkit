import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Protocol, runtime_checkable


@runtime_checkable
class AxisGenerator(Protocol):
    """Protocol for axis generators.

    Axis generators are responsible for generating the coordinate axis for NMR data dimensions.
    """

    def generate(self, size: int) -> np.ndarray:
        """Generate the axis values for a given size.

        Args:
            size: Number of points in the axis

        Returns:
            np.ndarray: Generated axis values
        """
        ...

    @property
    def is_uniform(self) -> bool:
        """Whether this generator produces uniform (equally spaced) values."""
        ...


@dataclass
class LinearGenerator(AxisGenerator):
    """Generator for linearly spaced axis values (uniform increment).

    This is suitable for most NMR time and frequency domains with regular sampling.
    """

    start: float = 0.0
    step: float = 1.0

    def __post_init__(self):
        """Validate parameters."""
        # Ensure step is a valid number
        if (
            not isinstance(self.step, (int, float))
            or np.isnan(self.step)
            or np.isinf(self.step)
        ):
            self.step = 1.0
        # Ensure start is a valid number
        if (
            not isinstance(self.start, (int, float))
            or np.isnan(self.start)
            or np.isinf(self.start)
        ):
            self.start = 0.0

    def generate(self, size: int) -> np.ndarray:
        """Generate linearly spaced values starting from 'start' with 'step' increment."""
        return np.arange(size, dtype=np.float64) * self.step + self.start

    @property
    def is_uniform(self) -> bool:
        """Linear generator always produces uniform spacing."""
        return True


@dataclass
class ExponentialGenerator(AxisGenerator):
    """Generator for exponentially spaced axis values.

    Useful for specialized experiments with logarithmic sampling.
    """

    start: float = 0.0
    growth_rate: float = 1.01

    def generate(self, size: int) -> np.ndarray:
        """Generate exponentially spaced values."""
        return (
            self.start
            + np.exp(np.arange(size, dtype=np.float64) * np.log(self.growth_rate))
            - 1.0
        )

    @property
    def is_uniform(self) -> bool:
        """Exponential generator produces non-uniform spacing."""
        return False


@dataclass
class NonUniformGenerator(AxisGenerator):
    """Generator for custom non-uniformly spaced axis values.

    Stores pre-computed axis values.
    """

    values: np.ndarray

    def generate(self, size: int) -> np.ndarray:
        """Return the pre-computed values.

        Args:
            size: Number of points to return

        Raises:
            ValueError: If requested size doesn't match stored values size
        """
        if size != len(self.values):
            raise ValueError(
                f"Requested size {size} doesn't match stored values size {len(self.values)}"
            )
        return self.values

    @property
    def is_uniform(self) -> bool:
        """Non-uniform generator produces non-uniform spacing."""
        return False


@dataclass
class DimensionInfo:
    """Metadata for a single NMR dimension.

    Attributes:
        size: Number of points in this dimension
        is_complex: Whether this dimension contains complex data (default: False)
        spectral_width: Spectral width in Hz (optional)
        observation_frequency: Observation frequency in MHz (optional)
        nucleus: Nucleus type (e.g., '1H', '13C') (optional)
        domain_type: Type of domain (free string) (optional)
        can_ft: Whether this dimension is Fourier transform capable (default: False)
        unit: Unit for spectral values (optional)
        transmitter_offset: Transmitter Frequency Offset in Hz (default: 0.0)
        axis_generator: Axis generator for creating coordinate values (default: Linear with step 0.0)
        domain_metadata: Dictionary for domain-specific parameters (optional)
    """

    size: int
    transmitter_offset: Optional[float] = None
    axis_generator: Optional[AxisGenerator] = None
    is_complex: bool = False
    spectral_width: Optional[float] = None
    observation_frequency: Optional[float] = None
    nucleus: Optional[str] = None
    domain_type: Optional[str] = None
    can_ft: bool = False
    unit: Optional[str] = None
    domain_metadata: Dict[str, Union[str, float, int, bool, np.ndarray]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Perform basic validation and set default values."""

        # Basic validation
        if self.size <= 0:
            raise ValueError("Dimension size must be positive")

        # Set default axis generator if none provided
        if self.axis_generator is None:
            self.axis_generator = LinearGenerator(step=0.0)

        # Set default unit based on domain type if not explicitly provided
        if self.unit is None:
            if self.domain_type == "time":
                self.unit = "s"  # Seconds for time domain
            elif self.domain_type == "frequency":
                # In NMR, frequency domain can use either Hz or ppm
                # For now, we'll leave it as None and let the user specify
                # This allows flexibility depending on the specific application
                pass
            else:
                # For other domain types, leave unit as None
                pass

    def generate_axis(self) -> np.ndarray:
        """Generate the axis values for this dimension.

        Returns:
            np.ndarray: Generated axis values
        """
        if self.axis_generator is None:
            raise ValueError("Axis generator not set")
        return self.axis_generator.generate(self.size)

    @property
    def increment(self) -> float:
        """Get the increment (step size) for this dimension.

        Returns:
            float: The increment value
        """
        if self.axis_generator is None:
            return 0.0
        return self.axis_generator.step


@dataclass
class NMRData:
    """Core NMR data structure for nmrkit.

    Attributes:
        data: Main data array. For standard data: shape = (dim1_size, dim2_size, ..., dimN_size).
              For hypercomplex data: shape = (dim1_size, dim2_size, ..., dimN_size, components).
        dimensions: List of dimension information objects
        metadata: Dictionary of general metadata (optional)
        source_format: Original format (e.g., 'nmrpipe', 'jeol') (optional)
        source_filename: Original filename (optional)
    """

    data: np.ndarray
    dimensions: List[DimensionInfo]
    metadata: Dict[str, Union[str, float, int, bool]] = field(default_factory=dict)
    source_format: Optional[str] = None
    source_filename: Optional[str] = None

    def __post_init__(self) -> None:
        # Validate data dimensions
        expected_ndim = len(self.dimensions)
        if self.data.ndim != expected_ndim:
            raise ValueError(f"Data has {
                    self.data.ndim} dimensions, expected {expected_ndim}")

        # Validate dimension sizes match data shape
        for i, (dim, size) in enumerate(
            zip(self.dimensions, self.data.shape[: len(self.dimensions)])
        ):
            if dim.size != size:
                raise ValueError(
                    f"Dimension {i + 1}: expected size {dim.size}, got {size}"
                )

    @property
    def ndim(self) -> int:
        """Number of NMR dimensions (excluding hypercomplex component dimension)."""
        return len(self.dimensions)

    @property
    def shape(self) -> tuple:
        """NMR dimensions shape (excluding hypercomplex component dimension)."""
        return self.data.shape[: len(self.dimensions)]

    @property
    def dtype(self) -> np.dtype:
        """Data type."""
        return self.data.dtype

    @property
    def is_complex(self) -> bool:
        """Check if data is complex."""
        return np.iscomplexobj(self.data)

    @property
    def full_shape(self) -> tuple:
        """Full data shape."""
        return self.data.shape

    def copy(self) -> "NMRData":
        """Create a deep copy of the NMRData object."""
        # Create new dimension info objects
        new_dimensions = [DimensionInfo(**vars(dim)) for dim in self.dimensions]

        return NMRData(
            data=self.data.copy(),
            dimensions=new_dimensions,
            metadata=self.metadata.copy(),
            source_format=self.source_format,
            source_filename=self.source_filename,
        )
