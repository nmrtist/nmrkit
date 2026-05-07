"""Processing workflows for nmrkit."""

from .basic_1d import process as basic_1d_process
from .basic_2d import process as basic_2d_process

__all__ = ["basic_1d_process", "basic_2d_process"]
