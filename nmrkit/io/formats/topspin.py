import os
import re
from typing import Dict, List, Optional
import numpy as np

from nmrkit.core import NMRData, DimensionInfo, LinearGenerator
from nmrkit.io.base import FormatReader
from nmrkit.utils import complexify


class TopSpinReader(FormatReader):
    """Reader for Bruker TopSpin NMR data format.

    This class handles reading Bruker TopSpin data files, including:
    - Binary data files (fid/ser)
    - Parameter files (acqus, acqu2s, etc.)

    The TopSpin format stores data in a directory structure with:
    - fid/ser: Binary data files
    - acqus: Acquisition parameters for direct dimension
    - acqu2s: Acquisition parameters for indirect dimensions
    - procs: Processing parameters (for processed data)
    """

    # Bruker data types mapping
    DATATYPES = {
        0: np.int32,
        2: np.float64,
    }

    # Endianness mapping
    ENDIANNESS = {
        0: "little",  # BYTORDA=0: little-endian
        1: "big",  # BYTORDA=1: big-endian
    }

    def __init__(self, filename: str, options: Optional[Dict] = None):
        """Initialize the TopSpin reader.

        Args:
            filename: Path to the TopSpin data directory or fid/ser file
            options: Dictionary of reader-specific options
        """
        super().__init__(filename, options)

        # Determine if we're reading from a directory, fid file, or ser file
        if os.path.isdir(filename):
            self._data_dir = filename
            self._fid_path = os.path.join(filename, "fid")
            self._ser_path = os.path.join(filename, "ser")
        else:
            self._data_dir = os.path.dirname(filename)
            if filename.endswith("fid"):
                self._fid_path = filename
                self._ser_path = os.path.join(self._data_dir, "ser")
            elif filename.endswith("ser"):
                self._ser_path = filename
                self._fid_path = os.path.join(self._data_dir, "fid")
            else:
                raise ValueError(
                    f"Invalid filename {filename}: must be a directory, fid file, or ser file"
                )

        self._parameters = {}
        self._dimensions = []
        self._data = None

    def read(self) -> NMRData:
        """Read Bruker TopSpin data and return an NMRData object.

        Returns:
            NMRData: An NMRData object containing the read data and metadata

        Raises:
            IOError: If there's an error reading the file
            ValueError: If the file format is invalid
        """
        # Read acquisition parameters
        self._read_parameters()

        # Determine data file to read (fid or ser)
        data_path = self._fid_path if os.path.exists(self._fid_path) else self._ser_path
        if not os.path.exists(data_path):
            raise IOError(f"No fid or ser file found in {self._data_dir}")

        # Read binary data
        self._read_binary_data(data_path)

        # Create dimension information
        dimensions = self._create_dimensions()

        # Create metadata dictionary
        metadata = self._create_metadata()

        return NMRData(
            data=self._data,
            dimensions=dimensions,
            metadata=metadata,
            source_format="topspin",
            source_filename=self.filename,
        )

    def _read_parameters(self) -> None:
        """Read acquisition parameters from acqus files.

        Reads parameters from acqus (direct dimension) and acqu2s, acqu3s, etc.
        (indirect dimensions) files.
        """
        # Read direct dimension parameters
        acqus_path = os.path.join(self._data_dir, "acqus")
        if os.path.exists(acqus_path):
            self._parameters["direct"] = self._parse_acqus_file(acqus_path)
        else:
            raise IOError(f"Missing acqus file in {self._data_dir}")

        # Read indirect dimension parameters
        dim = 2
        while True:
            acquNs_path = os.path.join(self._data_dir, f"acqu{dim}s")
            if os.path.exists(acquNs_path):
                self._parameters[f"indirect{dim}"] = self._parse_acqus_file(acquNs_path)
                dim += 1
            else:
                break

    def _parse_acqus_file(self, filepath: str) -> Dict[str, any]:
        """Parse a Bruker acqus parameter file.

        Args:
            filepath: Path to the acqus file

        Returns:
            Dict: Parsed parameters
        """
        params = {}

        with open(filepath, "r") as f:
            content = f.read()

        # Parse parameters using regex
        parameter_pattern = r"\$(\w+)=(.*)\n"
        matches = re.findall(parameter_pattern, content)

        for key, value in matches:
            # Clean up value
            value = value.strip()

            # Try to convert to appropriate type
            try:
                # Try integer first
                params[key] = int(value)
            except ValueError:
                try:
                    # Try float next
                    params[key] = float(value)
                except ValueError:
                    # Leave as string if conversion fails
                    params[key] = value

        return params

    def _read_binary_data(self, data_path: str) -> None:
        """Read binary data from fid or ser file.

        Args:
            data_path: Path to the binary data file
        """
        # Get parameters from direct dimension
        direct_params = self._parameters["direct"]

        # Determine data type and endianness
        dtype_code = direct_params.get("DTYPA", 2)  # Default to float64
        dtype_type = self.DATATYPES.get(dtype_code, np.float64)

        endian_code = direct_params.get("BYTORDA", 0)  # Default to little-endian
        endian = self.ENDIANNESS.get(endian_code, "little")
        endian_char = "<" if endian == "little" else ">"

        # For TopSpin, complex data is stored as interleaved real/imaginary components
        # We need to read it as real dtype first, then combine into complex
        if dtype_type == np.float64:
            # Complex float64: stored as interleaved float64 real and imaginary
            comp_dtype = np.dtype(endian_char + "f8")
            bytes_per_comp = 8
        elif dtype_type == np.int32:
            # Complex int32: stored as interleaved int32 real and imaginary
            comp_dtype = np.dtype(endian_char + "i4")
            bytes_per_comp = 4
        else:
            # Apply endianness to regular dtype
            dtype_str = dtype_type.str
            if endian == "big" and dtype_str[0] == "<":
                dtype_str = ">" + dtype_str[1:]
            elif endian == "little" and dtype_str[0] == ">":
                dtype_str = "<" + dtype_str[1:]
            comp_dtype = np.dtype(dtype_str)
            bytes_per_comp = comp_dtype.itemsize

        bytes_per_point = bytes_per_comp * 2  # Each complex point has real + imaginary

        # Determine number of dimensions
        ndim = 1 + len([k for k in self._parameters if k.startswith("indirect")])

        # Get file size
        file_size = os.path.getsize(data_path)

        # Read the binary data
        with open(data_path, "rb") as f:
            raw_data = f.read()

        # Convert to numpy array as real components first
        data = np.frombuffer(raw_data, dtype=comp_dtype)

        # Calculate actual data size
        actual_size = data.size

        # For complex data, we need to reshape and combine real/imaginary parts
        if dtype_type in [np.float64, np.int32]:
            # Check if data size is even (required for complex data)
            if actual_size % 2 != 0:
                # Truncate if odd size (shouldn't happen with proper Bruker
                # files)
                data = data[:-1]
                actual_size = data.size

            # Use complexify to convert interleaved real/imaginary to complex
            data = complexify(data, mode="interleaved", first_component="real")

        # Reshape data based on dimensions
        if ndim == 1:
            # 1D data: just return as 1D array
            self._data = data
        elif ndim == 2:
            # Get TD values from parameters
            try:
                td0 = direct_params.get("TD", 1024)  # Direct dimension points
                td1 = self._parameters["indirect2"].get(
                    "TD", 1
                )  # Indirect dimension points
            except KeyError:
                td0 = 1024
                td1 = 1

            # Calculate expected points per increment, considering padding
            # Bruker pads data to 1024-byte blocks
            bytes_per_block = 1024
            # Each complex point is bytes_per_point
            points_per_block = bytes_per_block // bytes_per_point
            expected_points = int(np.ceil(td0 / points_per_block) * points_per_block)

            # Try different reshape strategies
            try:
                # First try: use TD values directly
                # For NMRData, direct dimension (F2) should be first, so
                # reshape as (direct, indirect)
                self._data = data.reshape(td1, expected_points).T
            except ValueError:
                try:
                    # Second try: use actual data size
                    if actual_size % td1 == 0:
                        self._data = data.reshape(td1, actual_size // td1).T
                    else:
                        # Third try: use TD0 from parameters directly
                        self._data = data.reshape(td1, td0).T
                except ValueError:
                    # Final try: calculate based on file size
                    calculated_points = file_size // (td1 * bytes_per_point)
                    if calculated_points > 0:
                        self._data = data.reshape(td1, calculated_points).T
                    else:
                        # If all else fails, raise informative error
                        raise ValueError(
                            f"Cannot reshape data of size {actual_size} for 2D experiment. "
                            f"File size: {file_size} bytes, bytes per point: {bytes_per_point}, "
                            f"Indirect dimension points: {td1}"
                        )
        else:
            # For higher dimensions, we'd need to handle ser file format properly
            # This is a simplified implementation
            raise NotImplementedError(
                "Reading data with more than 2 dimensions is not implemented yet"
            )

    def _create_dimensions(self) -> List[DimensionInfo]:
        """Create dimension information from parameters.

        Returns:
            List[DimensionInfo]: List of dimension information objects
        """
        dimensions = []

        # Get number of dimensions
        ndim = 1 + len([k for k in self._parameters if k.startswith("indirect")])

        # For 2D data, direct dimension (F2) should be first in NMRData, then indirect (F1)
        # This matches the transposed data shape which is (direct, indirect)
        if ndim == 2:
            # Add direct dimension (F2) first
            direct_params = self._parameters["direct"]
            is_complex = True
            spectral_width = direct_params.get("SW_h", 0.0)
            # Ensure spectral_width is a valid number and not NaN
            if not isinstance(spectral_width, (int, float)) or np.isnan(spectral_width):
                spectral_width = 0.0

            # Get observation frequency and nucleus
            sf = direct_params.get("SFO1", 0.0)
            nucleus = direct_params.get("NUC1", "").strip()

            # Determine direct dimension size from actual data
            if self._data is not None and len(self._data.shape) > 1:
                # For 2D, direct dimension is now the first axis after
                # transpose
                size = self._data.shape[0]
            else:
                # Fallback to TD from parameters
                td = direct_params.get("TD", 1)
                size = td // 2  # Convert from real points to complex points

            dim = DimensionInfo(
                size=size,
                is_complex=is_complex,
                spectral_width=spectral_width,
                observation_frequency=sf,
                nucleus=nucleus,
                domain_type="time",
                can_ft=True,
                unit="Hz",
                transmitter_offset=direct_params.get(
                    "O1", 0.0
                ),  # O1 is transmitter offset in Hz
                axis_generator=LinearGenerator(
                    step=1.0 / (spectral_width if spectral_width > 0.0 else 1.0)
                ),
                domain_metadata={"name": "F2"},
            )
            dimensions.append(dim)

            # Add indirect dimension (F1) second
            indirect_params = self._parameters.get("indirect2", {})
            # Determine indirect dimension size from actual data
            if self._data is not None and len(self._data.shape) > 1:
                # For 2D, indirect dimension is now the second axis after
                # transpose
                size = self._data.shape[1]
            else:
                # Fallback to TD from parameters
                size = indirect_params.get("TD", 1)
            is_complex = False  # Indirect dimensions are usually real
            spectral_width = indirect_params.get("SW_h", 0.0)
            # Ensure spectral_width is a valid number and not NaN
            if not isinstance(spectral_width, (int, float)) or np.isnan(spectral_width):
                spectral_width = 0.0

            # Get observation frequency and nucleus
            sf = indirect_params.get("SFO1", 0.0)
            nucleus = indirect_params.get("NUC1", "").strip()

            dim = DimensionInfo(
                size=size,
                is_complex=is_complex,
                spectral_width=spectral_width,
                observation_frequency=sf,
                nucleus=nucleus,
                domain_type="time",
                can_ft=True,
                unit="Hz",
                transmitter_offset=indirect_params.get(
                    "O1", 0.0
                ),  # O1 is transmitter offset in Hz
                axis_generator=LinearGenerator(
                    step=1.0 / (spectral_width if spectral_width > 0.0 else 1.0)
                ),
                domain_metadata={"name": "F1"},
            )
            dimensions.append(dim)
        else:
            # For 1D data or higher dimensions, use original order
            # Add direct dimension (F2)
            direct_params = self._parameters["direct"]
            is_complex = True
            spectral_width = direct_params.get("SW_h", 0.0)
            # Ensure spectral_width is a valid number and not NaN
            if not isinstance(spectral_width, (int, float)) or np.isnan(spectral_width):
                spectral_width = 0.0

            # Get observation frequency and nucleus
            sf = direct_params.get("SFO1", 0.0)
            nucleus = direct_params.get("NUC1", "").strip()

            # Determine direct dimension size from actual data
            if self._data is not None:
                if len(self._data.shape) == 1:
                    # 1D data
                    size = self._data.shape[0]
                else:
                    # For higher dimensions, direct dimension is the last axis
                    size = self._data.shape[-1]
            else:
                # Fallback to TD from parameters
                td = direct_params.get("TD", 1)
                size = td // 2  # Convert from real points to complex points

            dim = DimensionInfo(
                size=size,
                is_complex=is_complex,
                spectral_width=spectral_width,
                observation_frequency=sf,
                nucleus=nucleus,
                domain_type="time",
                can_ft=True,
                unit="Hz",
                transmitter_offset=direct_params.get(
                    "O1", 0.0
                ),  # O1 is transmitter offset in Hz
                axis_generator=LinearGenerator(
                    step=1.0 / (spectral_width if spectral_width > 0.0 else 1.0)
                ),
                domain_metadata={"name": "F2"},
            )
            dimensions.append(dim)

            # Add other indirect dimensions (F3, etc.) if any
            for i in range(
                3, 2 + len([k for k in self._parameters if k.startswith("indirect")])
            ):
                indirect_params = self._parameters[f"indirect{i}"]
                # Determine indirect dimension size from actual data
                if self._data is not None and len(self._data.shape) >= i - 1:
                    # For higher dimensions, indirect dimensions are earlier
                    # axes
                    size = self._data.shape[i - 2]
                else:
                    # Fallback to TD from parameters
                    size = indirect_params.get("TD", 1)
                is_complex = False  # Indirect dimensions are usually real
                spectral_width = indirect_params.get("SW_h", 0.0)
                # Ensure spectral_width is a valid number and not NaN
                if not isinstance(spectral_width, (int, float)) or np.isnan(
                    spectral_width
                ):
                    spectral_width = 0.0

                # Get observation frequency and nucleus
                sf = indirect_params.get("SFO1", 0.0)
                nucleus = indirect_params.get("NUC1", "").strip()

                dim = DimensionInfo(
                    size=size,
                    is_complex=is_complex,
                    spectral_width=spectral_width,
                    observation_frequency=sf,
                    nucleus=nucleus,
                    domain_type="time",
                    can_ft=True,
                    unit="Hz",
                    transmitter_offset=indirect_params.get(
                        "O1", 0.0
                    ),  # O1 is transmitter offset in Hz
                    axis_generator=LinearGenerator(
                        step=1.0 / (spectral_width if spectral_width > 0.0 else 1.0)
                    ),
                    domain_metadata={"name": f"F{i - 1}"},
                )
                dimensions.append(dim)

        return dimensions

    def _create_metadata(self) -> Dict[str, any]:
        """Create metadata dictionary from parameters.

        Returns:
            Dict: Metadata dictionary
        """
        metadata = {
            "source_format": "topspin",
            "source_filename": self.filename,
            "data_directory": self._data_dir,
            "parameters": self._parameters,
        }

        return metadata
