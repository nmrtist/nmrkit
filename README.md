# nmrkit

[![PyPI version](https://badge.fury.io/py/nmrkit.svg)](https://pypi.org/project/nmrkit/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Poetry](https://img.shields.io/badge/Poetry-managed-blueviolet)](https://python-poetry.org/)

nmrkit is a Python library for nuclear magnetic resonance (NMR) data processing and analysis.

This is an early-stage project under active development. Contributions are welcome to help expand its capabilities!

## Features

- **Data Import/Export**: Support for common NMR formats including TopSpin and Delta
- **Basic Processing**: Fourier transform, apodization (exponential multiplication), zero filling, phase correction
- **Acquisition Planning**: Calculate DOSY diffusion experiment timing and gradient settings
- **Visualization**: Interactive plotting with customizable parameters
- **Simple API**: Intuitive functions for data processing workflows

## Requirements

- Python >= 3.12
- NumPy
- SciPy
- Matplotlib (optional, for visualization)

## Installation

### From PyPI (Recommended)

```bash
pip install nmrkit
```

With visualization support:

```bash
pip install nmrkit[visualization]
```

## Quick Start

### Automatic Processing

Use the `auto_process` function for automated application of common processing steps:

```python
import nmrkit as nk

# Load NMR data
data = nk.read('path/to/data')

# Automatically process the spectrum
data = nk.auto_process(data)

# Save the plot to a PDF file
nk.plot(data, output_path="spectrum.pdf")
```

### Manual Processing

For more control, use individual processing functions:

```python
import nmrkit as nk

# Load data with explicit format specification
data = nk.read('path/to/data.jdf', format='delta')

# Apply exponential multiplication (apodization) with 1.0 Hz line broadening
data = nk.em(data, lb=1.0)

# Zero fill to 2048 points for improved resolution
data = nk.zf(data, size=2048)

# Perform Fourier transform
data = nk.ft(data)

# Apply phase correction
data = nk.phase(data, ph0=10.0, ph1=25.0)

# Plot the manually processed spectrum
nk.plot(data)
```

### DOSY Acquisition Planning

Use acquisition helpers to estimate DOSY experiment settings from a diffusion
coefficient and the maximum gradient strength of an instrument:

```python
dosy_params = nk.calculate_dosy_settings(
    diffusion_coefficient_m2_s=5.8e-10,
    max_gradient_t_m=0.535,
    target_attenuation=0.05,
)

print(f"Diffusion time (big delta):   {dosy_params.diffusion_time_ms:.2f} ms")
print(f"Gradient duration (little delta): {dosy_params.gradient_duration_ms:.4f} ms")
print(f"Achieved residual signal:     {dosy_params.achieved_attenuation:.4f}")
```

## License

nmrkit is released under the Apache License 2.0. See the [LICENSE](LICENSE) file for more information.

## Contributing

The nmrkit project welcomes your expertise and enthusiasm!

### Reporting Issues

If you encounter a bug or have an idea for a new feature, please open an issue on GitHub. When reporting bugs, please include:
- A clear description of the issue
- Steps to reproduce (if applicable)
- Expected vs actual behavior
- Your Python version and operating system

### Contributing Code

To contribute code to nmrkit:

1. **Fork and Clone**: Fork the repository and clone it locally
2. **Set up Development Environment**: 
   ```bash
   poetry install
   ```
3. **Create a Branch**: Create a new branch for your feature or bug fix
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make Changes**: Implement your changes following the existing code style
5. **Run Tests**: Ensure all tests pass
   ```bash
   poetry run pytest
   ```
6. **Commit**: Commit your changes with clear, descriptive messages
7. **Push and Submit PR**: Push to your fork and submit a pull request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints for new functions
- Include docstrings following the existing format
- Add tests for new functionality
- Ensure all tests pass before submitting PR

Small improvements or fixes are always appreciated. Even minor contributions can make a significant difference!

## Contact

- **GitHub Repository**: https://github.com/nmrtist/nmrkit
