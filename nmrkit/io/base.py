import abc
import os
from typing import Dict, Optional, Union
from nmrkit.core import NMRData


class FormatReader(abc.ABC):
    """Abstract base class for NMR data format readers.

    This class defines the common interface for all format readers in nmrkit.
    All format-specific readers must inherit from this class and implement
    the abstract methods.

    Attributes:
        filename: Path to the input file
        options: Dictionary of reader-specific options
    """

    def __init__(self, filename: str, options: Optional[Dict] = None):
        """Initialize the reader.

        Args:
            filename: Path to the input file
            options: Dictionary of reader-specific options
        """
        self.filename = filename
        self.options = options or {}

    @abc.abstractmethod
    def read(self) -> NMRData:
        """Read data from the file and return an NMRData object.

        Returns:
            NMRData: An NMRData object containing the read data and metadata

        Raises:
            IOError: If there's an error reading the file
            ValueError: If the file format is invalid
        """
        pass


class FormatWriter(abc.ABC):
    """Abstract base class for NMR data format writers.

    This class defines the common interface for all format writers in nmrkit.
    All format-specific writers must inherit from this class and implement
    the abstract methods.

    Attributes:
        filename: Path to the output file
        options: Dictionary of writer-specific options
    """

    def __init__(self, filename: str, options: Optional[Dict] = None):
        """Initialize the writer.

        Args:
            filename: Path to the output file
            options: Dictionary of writer-specific options
        """
        self.filename = filename
        self.options = options or {}

    @abc.abstractmethod
    def write(self, data: NMRData):
        """Write NMRData to the file.

        Args:
            data: NMRData object to write to file

        Raises:
            IOError: If there's an error writing to the file
            ValueError: If the data format is not compatible with the writer
        """
        pass


def read(
    filename: str, format: Optional[str] = None, options: Optional[Dict] = None
) -> NMRData:
    """Convenience function to read NMR data from file.

    This function automatically detects the file format or uses the specified format,
    creates the appropriate reader instance, and reads the data.

    Args:
        filename: Path to the input file
        format: Optional format specification (e.g., 'delta', 'nmrpipe', 'topspin')
        options: Dictionary of reader-specific options

    Returns:
        NMRData: An NMRData object containing the read data and metadata

    Raises:
        IOError: If there's an error reading the file
        ValueError: If the file format is unknown or invalid
    """
    # Import here to avoid circular imports
    from nmrkit.io.formats import DeltaReader, TopSpinReader

    # If format is not specified, try to detect it from the file
    if format is None:
        if os.path.isdir(filename):
            # Check if directory contains TopSpin-specific files
            if any(
                os.path.exists(os.path.join(filename, f))
                for f in ["acqus", "fid", "ser"]
            ):
                format = "topspin"
        elif os.path.isfile(filename):
            # Check file extension for format detection
            file_ext = filename.lower()
            if file_ext.endswith(".jdf"):
                format = "delta"  # JDF files use Delta format
            elif file_ext.endswith(("fid", "ser")):
                format = "topspin"  # fid and ser files use TopSpin format

        # If format still not detected, raise error
        if format is None:
            raise ValueError(
                f"File format could not be automatically detected for '{filename}'. Please specify the format explicitly using the format parameter."
            )

    # Create the appropriate reader instance
    if format.lower() == "delta":
        reader = DeltaReader(filename, options)
    elif format.lower() == "topspin":
        reader = TopSpinReader(filename, options)
    # Add support for other formats here
    # elif format.lower() == 'nmrpipe':
    #     reader = NMRPipeReader(filename, options)
    # elif format.lower() == 'vnmrj':
    #     reader = VnmrJReader(filename, options)
    else:
        raise ValueError(f"Unknown format: {format}")

    # Read and return the data
    return reader.read()


def write(
    filename: str,
    data: NMRData,
    format: Optional[str] = None,
    options: Optional[Dict] = None,
):
    """Convenience function to write NMR data to file.

    This function uses the specified format to create the appropriate writer instance,
    and writes the data to file.

    Args:
        filename: Path to the output file
        data: NMRData object to write to file
        format: Optional format specification (e.g., 'delta', 'nmrpipe', 'topspin')
        options: Dictionary of writer-specific options

    Raises:
        IOError: If there's an error writing to the file
        ValueError: If the format is unknown or data is incompatible with the writer
    """
    # Import here to avoid circular imports
    # from .formats import DeltaWriter, NMRPipeWriter, TopspinWriter, VnmrJWriter

    # If format is not specified, try to detect it from the filename extension
    if format is None:
        # Simple format detection based on file extension
        # This should be extended with more sophisticated detection
        format = "delta"  # Default to Delta for now

    # Create the appropriate writer instance
    if format.lower() == "delta":
        # writer = DeltaWriter(filename, options)
        raise NotImplementedError("Delta writer not yet implemented")
    # Add support for other formats here
    # elif format.lower() == 'nmrpipe':
    #     writer = NMRPipeWriter(filename, options)
    # elif format.lower() == 'topspin':
    #     writer = TopspinWriter(filename, options)
    # elif format.lower() == 'vnmrj':
    #     writer = VnmrJWriter(filename, options)
    else:
        raise ValueError(f"Unknown format: {format}")

    # Write the data
    # writer.write(data)
