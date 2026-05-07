"""Core functionality for nmrkit."""

from .data import (
    DimensionInfo,
    NMRData,
    LinearGenerator,
    ExponentialGenerator,
    NonUniformGenerator,
)

__all__ = [
    "DimensionInfo",
    "NMRData",
    "LinearGenerator",
    "ExponentialGenerator",
    "NonUniformGenerator",
]
