"""Utilities for planning DOSY diffusion experiment parameters."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, sqrt

import numpy as np

from nmrkit.constants import (
    DEFAULT_DOSY_TARGET_ATTENUATION,
    GYROMAGNETIC_RATIOS_RAD_S_T,
)


# ---------------------------------------------------------------------------
# Constants for parameter recommendation
# ---------------------------------------------------------------------------
DEFAULT_DIFFUSION_TIME_S = 0.100
MAX_GRADIENT_DURATION_S = 0.005
MAX_DIFFUSION_TIME_S = 0.200


@dataclass(frozen=True)
class DOSYSettings:
    """Calculated DOSY acquisition settings in SI units."""

    diffusion_coefficient_m2_s: float
    max_gradient_t_m: float
    diffusion_time_s: float
    gradient_duration_s: float
    target_attenuation: float
    achieved_attenuation: float
    gamma_rad_s_t: float

    @property
    def diffusion_time_ms(self) -> float:
        """Diffusion time, big delta, in milliseconds."""
        return self.diffusion_time_s * 1000.0

    @property
    def gradient_duration_ms(self) -> float:
        """Diffusion gradient duration, little delta, in milliseconds."""
        return self.gradient_duration_s * 1000.0

    @property
    def diffusion_weighting_s_m2(self) -> float:
        """Diffusion weighting b-value for the achieved parameters."""
        return diffusion_weighting(
            gradient_t_m=self.max_gradient_t_m,
            diffusion_time_s=self.diffusion_time_s,
            gradient_duration_s=self.gradient_duration_s,
            gamma_rad_s_t=self.gamma_rad_s_t,
        )


def gamma_for_nucleus(nucleus: str = "1H") -> float:
    """Return the gyromagnetic ratio for a nucleus in rad s^-1 T^-1."""
    try:
        return GYROMAGNETIC_RATIOS_RAD_S_T[nucleus]
    except KeyError as exc:
        supported = ", ".join(sorted(GYROMAGNETIC_RATIOS_RAD_S_T))
        raise ValueError(
            f"Unsupported nucleus {nucleus!r}; supported: {supported}"
        ) from exc


def diffusion_weighting(
    *,
    gradient_t_m: float,
    diffusion_time_s: float,
    gradient_duration_s: float,
    gamma_rad_s_t: float | None = None,
    nucleus: str = "1H",
) -> float:
    """Calculate Stejskal-Tanner diffusion weighting (b-value).

    The returned value is the factor multiplied by the diffusion coefficient in
    ``I / I0 = exp(-D * b)``.
    """
    gamma = gamma_for_nucleus(nucleus) if gamma_rad_s_t is None else gamma_rad_s_t
    _validate_positive("gradient_t_m", gradient_t_m)
    _validate_positive("diffusion_time_s", diffusion_time_s)
    _validate_positive("gradient_duration_s", gradient_duration_s)
    _validate_positive("gamma_rad_s_t", abs(gamma))

    effective_time = diffusion_time_s - gradient_duration_s / 3.0
    if effective_time <= 0.0:
        raise ValueError(
            "diffusion_time_s must be greater than gradient_duration_s / 3"
        )

    return (gamma * gradient_t_m * gradient_duration_s) ** 2 * effective_time


def attenuation_factor(
    *,
    diffusion_coefficient_m2_s: float,
    gradient_t_m: float,
    diffusion_time_s: float,
    gradient_duration_s: float,
    gamma_rad_s_t: float | None = None,
    nucleus: str = "1H",
) -> float:
    """Calculate the residual signal fraction ``I / I0`` for a DOSY point."""
    _validate_positive("diffusion_coefficient_m2_s", diffusion_coefficient_m2_s)
    b_value = diffusion_weighting(
        gradient_t_m=gradient_t_m,
        diffusion_time_s=diffusion_time_s,
        gradient_duration_s=gradient_duration_s,
        gamma_rad_s_t=gamma_rad_s_t,
        nucleus=nucleus,
    )
    return float(np.exp(-diffusion_coefficient_m2_s * b_value))


def calculate_dosy_settings(
    *,
    diffusion_coefficient_m2_s: float,
    max_gradient_t_m: float,
    target_attenuation: float = DEFAULT_DOSY_TARGET_ATTENUATION,
    nucleus: str = "1H",
    gamma_rad_s_t: float | None = None,
) -> DOSYSettings:
    """Recommend practical DOSY timing parameters for a target attenuation.

    The algorithm follows the reference recommendation that the highest gradient
    point should leave roughly *target_attenuation* residual signal (default 5%).

    Parameters
    ----------
    diffusion_coefficient_m2_s : float
        Estimated diffusion coefficient in m^2/s.
    max_gradient_t_m : float
        Maximum available gradient strength in T/m.
    target_attenuation : float, optional
        Desired residual signal fraction at maximum gradient (default 0.05).
    nucleus : str, optional
        Nucleus identifier (default "1H").
    gamma_rad_s_t : float, optional
        Gyromagnetic ratio in rad s^-1 T^-1. If None, looked up from *nucleus*.

    Returns
    -------
    DOSYSettings
        Recommended diffusion time and gradient duration.

    Raises
    ------
    ValueError
        If no valid parameter combination can be found within physical bounds.
    """
    _validate_positive("diffusion_coefficient_m2_s", diffusion_coefficient_m2_s)
    _validate_positive("max_gradient_t_m", max_gradient_t_m)
    _validate_attenuation(target_attenuation)

    gamma = gamma_for_nucleus(nucleus) if gamma_rad_s_t is None else gamma_rad_s_t
    _validate_positive("gamma_rad_s_t", abs(gamma))

    target_exponent = -log(target_attenuation)

    # Try default diffusion time first (100 ms)
    diffusion_time_s = DEFAULT_DIFFUSION_TIME_S
    gradient_duration_s = _solve_gradient_duration(
        target_exponent=target_exponent,
        diffusion_coefficient_m2_s=diffusion_coefficient_m2_s,
        max_gradient_t_m=max_gradient_t_m,
        diffusion_time_s=diffusion_time_s,
        gamma_rad_s_t=gamma,
    )

    # If gradient duration exceeds 5 ms, increase diffusion time
    if gradient_duration_s > MAX_GRADIENT_DURATION_S:
        diffusion_time_s, gradient_duration_s = _adjust_diffusion_time(
            target_exponent=target_exponent,
            diffusion_coefficient_m2_s=diffusion_coefficient_m2_s,
            max_gradient_t_m=max_gradient_t_m,
            gamma_rad_s_t=gamma,
        )

    achieved = attenuation_factor(
        diffusion_coefficient_m2_s=diffusion_coefficient_m2_s,
        gradient_t_m=max_gradient_t_m,
        diffusion_time_s=diffusion_time_s,
        gradient_duration_s=gradient_duration_s,
        gamma_rad_s_t=gamma,
    )

    return DOSYSettings(
        diffusion_coefficient_m2_s=diffusion_coefficient_m2_s,
        max_gradient_t_m=max_gradient_t_m,
        diffusion_time_s=diffusion_time_s,
        gradient_duration_s=gradient_duration_s,
        target_attenuation=target_attenuation,
        achieved_attenuation=achieved,
        gamma_rad_s_t=gamma,
    )


def _solve_gradient_duration(
    *,
    target_exponent: float,
    diffusion_coefficient_m2_s: float,
    max_gradient_t_m: float,
    diffusion_time_s: float,
    gamma_rad_s_t: float,
) -> float:
    """Solve for gradient duration given a fixed diffusion time.

    From the Stejskal-Tanner equation:
        -ln(target) = D * (gamma * G * delta)^2 * (Delta - delta/3)

    This is a cubic in delta:
        delta^2 * (Delta - delta/3) = target_exponent / (D * (gamma * G)^2)

    Rearranged:
        -1/3 * delta^3 + Delta * delta^2 - C = 0
    where C = target_exponent / (D * (gamma * G)^2).
    """
    coefficient = (
        diffusion_coefficient_m2_s * (gamma_rad_s_t * max_gradient_t_m) ** 2
    )
    required_timing = target_exponent / coefficient

    # Cubic: -1/3 * d^3 + Delta * d^2 - required_timing = 0
    roots = np.roots([-1.0 / 3.0, diffusion_time_s, 0.0, -required_timing])
    valid_roots = [
        float(root.real)
        for root in roots
        if abs(root.imag) < 1e-10 and 0.0 < root.real < 3.0 * diffusion_time_s
    ]
    if not valid_roots:
        raise ValueError(
            "No positive gradient duration satisfies the requested attenuation "
            f"for diffusion_time_s={diffusion_time_s}"
        )
    return min(valid_roots)


def _adjust_diffusion_time(
    *,
    target_exponent: float,
    diffusion_coefficient_m2_s: float,
    max_gradient_t_m: float,
    gamma_rad_s_t: float,
) -> tuple[float, float]:
    """Increase diffusion time until gradient duration <= 5 ms.

    If diffusion time exceeds 200 ms, a warning message is included in the
    returned tuple (caller should warn the user that Delta should not exceed T1).
    """
    coefficient = (
        diffusion_coefficient_m2_s * (gamma_rad_s_t * max_gradient_t_m) ** 2
    )
    required_timing = target_exponent / coefficient

    # We need: delta <= 5 ms and Delta > delta / 3
    # From the equation: Delta = required_timing / delta^2 + delta / 3
    # To minimize Delta for a given required_timing, we can use calculus:
    # d(Delta)/d(delta) = -2*required_timing/delta^3 + 1/3 = 0
    # => delta^3 = 6 * required_timing => delta = (6*required_timing)^(1/3)
    optimal_delta = (6.0 * required_timing) ** (1.0 / 3.0)

    if optimal_delta <= MAX_GRADIENT_DURATION_S:
        # Use the optimal point where gradient duration is minimized
        delta = optimal_delta
        delta = min(delta, MAX_GRADIENT_DURATION_S)
    else:
        # Even the optimal delta exceeds 5 ms, so cap at 5 ms
        delta = MAX_GRADIENT_DURATION_S

    delta = float(delta)
    delta = max(delta, 1e-6)  # numerical safety
    diffusion_time_s = required_timing / (delta ** 2) + delta / 3.0

    if diffusion_time_s > MAX_DIFFUSION_TIME_S:
        import warnings

        warnings.warn(
            f"Recommended diffusion time ({diffusion_time_s * 1000:.1f} ms) exceeds "
            f"{MAX_DIFFUSION_TIME_S * 1000:.0f} ms. Ensure diffusion time does not "
            "exceed the sample's longitudinal relaxation time T1.",
            UserWarning,
            stacklevel=3,
        )

    return diffusion_time_s, delta


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_attenuation(value: float) -> None:
    if not isfinite(value) or value <= 0.0 or value >= 1.0:
        raise ValueError("target_attenuation must be > 0 and < 1")


def _validate_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be > 0")
