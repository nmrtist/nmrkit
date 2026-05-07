"""Plotting functions for NMR data visualization."""

from typing import Optional

import numpy as np

from nmrkit.core import NMRData


def plot(data: NMRData, output_path: Optional[str] = None):
    """
    Plot NMR data.

    Args:
        data: NMR data to plot.
        output_path: Optional path to save the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install nmrkit[visualization]"
        )

    if data.ndim == 1:
        _plot_1d(data, output_path, plt)
    elif data.ndim == 2:
        _plot_2d(data, output_path, plt)
    else:
        raise NotImplementedError(
            f"Only 1D and 2D plotting are implemented currently, got {
                data.ndim}D data"
        )


def _plot_1d(data: NMRData, output_path: Optional[str] = None, plt=None):
    """
    Plot 1D NMR data.

    Args:
        data: 1D NMR data to plot.
        output_path: Optional path to save the plot.
        plt: matplotlib.pyplot instance.
    """
    dim = data.dimensions[0]
    signal = data.data.real

    is_frequency_domain = dim.domain_type == "frequency"

    x = dim.generate_axis()

    if is_frequency_domain:
        if dim.observation_frequency:
            x = x / dim.observation_frequency
            # Update unit to ppm after conversion
            dim.unit = "ppm"

        x = np.flip(x)
        signal = np.flip(signal)

    plt.figure(figsize=(11.69, 8.27))
    plt.plot(x, signal)

    if dim.unit:
        plt.xlabel(f"{dim.unit}", horizontalalignment="right", x=1.0)
    plt.ylabel("Intensity")
    plt.xlim(x[0], x[-1])

    y_min, y_max = signal.min(), signal.max()
    y_margin = (y_max - y_min) * 0.05  # 5% margin on top and bottom
    plt.ylim(y_min - y_margin, y_max + y_margin)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def _plot_2d(data: NMRData, output_path: Optional[str] = None, plt=None):
    """
    Plot 2D NMR data as contour plot.

    Args:
        data: 2D NMR data to plot.
        output_path: Optional path to save the plot.
        plt: matplotlib.pyplot instance.
    """
    # Get dimensions
    dim1 = data.dimensions[0]  # First dimension (now x-axis)
    dim2 = data.dimensions[1]  # Second dimension (now y-axis)

    # Get real part of the data
    signal = data.data.real

    # Generate axes
    x = dim1.generate_axis()  # First dimension (x-axis)
    y = dim2.generate_axis()  # Second dimension (y-axis)

    # Check if in frequency domain and convert to ppm if needed
    is_frequency_domain_f1 = dim1.domain_type == "frequency"
    is_frequency_domain_f2 = dim2.domain_type == "frequency"

    # Convert to ppm if in frequency domain and observation frequency is
    # available
    if is_frequency_domain_f1 and dim1.observation_frequency:
        x = x / dim1.observation_frequency
        # Update unit to ppm after conversion
        dim1.unit = "ppm"
        x = np.flip(x)
        signal = np.flip(signal, axis=0)

    if is_frequency_domain_f2 and dim2.observation_frequency:
        y = y / dim2.observation_frequency
        # Update unit to ppm after conversion
        dim2.unit = "ppm"
        y = np.flip(y)
        signal = np.flip(signal, axis=1)

    # Create figure
    plt.figure(figsize=(11.69, 8.27))

    # Calculate reasonable contour levels
    signal_range = signal.max() - signal.min()

    if signal_range > 0:
        # Try different approaches based on signal characteristics
        if signal.min() < 0:
            # If there are negative values, use symmetric levels around zero
            max_abs = max(abs(signal.min()), abs(signal.max()))
            levels = np.linspace(-max_abs * 0.9, max_abs * 0.9, 40)
        else:
            # If all positive, use percentile-based levels
            levels = np.percentile(signal, np.linspace(10, 100, 40))

        plt.contour(x, y, signal, levels=levels, linewidths=0.5)
    else:
        # If no signal range, create a simple plot
        levels = np.linspace(signal.min() - 1, signal.max() + 1, 10)
        plt.contour(x, y, signal, levels=levels, cmap="viridis")

    # Simplify labels to only show units and adjust positions
    if dim1.unit:
        plt.xlabel(dim1.unit, horizontalalignment="right", x=1.0)
    if dim2.unit:
        plt.ylabel(dim2.unit, horizontalalignment="left", y=0.95)

    # Set limits
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
