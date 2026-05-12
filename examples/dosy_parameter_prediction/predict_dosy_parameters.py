"""
Example: DOSY parameter prediction for aqueous samples.

This script demonstrates how to use ``nmrkit`` to recommend optimal
Diffusion time (big delta) and gradient duration (little delta) for a
DOSY experiment, given an estimated diffusion coefficient and the
maximum available gradient strength.

Reference sample
----------------
- Diffusion coefficient: water at ~25 C  -> 2.3e-5 cm^2/s
- Maximum gradient:      typical z-gradient coil -> 0.3 T/m

Usage
-----
    python predict_dosy_parameters.py

The script prints a formatted report with the recommended parameters
and the expected residual signal fraction at maximum gradient.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the package root is on sys.path when running the example directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import nmrkit as nk


def main() -> None:
    """Run the DOSY parameter prediction example."""
    # ------------------------------------------------------------------
    # User inputs
    # ------------------------------------------------------------------
    # Water diffusion coefficient at room temperature (~25 C)
    diffusion_coefficient_cm2_s = 2.3e-5

    # Convert to SI units (m^2/s)
    diffusion_coefficient_m2_s = diffusion_coefficient_cm2_s * 1e-4

    # Maximum gradient strength of the probe/system
    max_gradient_t_m = 0.3

    # Target residual signal at maximum gradient (default 5%)
    target_attenuation = 0.05

    # ------------------------------------------------------------------
    # Parameter recommendation
    # ------------------------------------------------------------------
    settings = nk.calculate_dosy_settings(
        diffusion_coefficient_m2_s=diffusion_coefficient_m2_s,
        max_gradient_t_m=max_gradient_t_m,
        target_attenuation=target_attenuation,
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("=" * 60)
    print("DOSY Parameter Prediction Report")
    print("=" * 60)
    print(f"{'Sample diffusion coefficient:':<40} {diffusion_coefficient_cm2_s:.2e} cm^2/s")
    print(f"{'Maximum gradient strength:':<40} {max_gradient_t_m:.1f} T/m")
    print(f"{'Target residual signal (I/I0):':<40} {target_attenuation:.0%}")
    print("-" * 60)
    print(f"{'Recommended diffusion time (Delta):':<40} {settings.diffusion_time_ms:.2f} ms")
    print(f"{'Recommended gradient duration (delta):':<40} {settings.gradient_duration_ms:.2f} ms")
    print(f"{'Achieved residual signal (I/I0):':<40} {settings.achieved_attenuation:.4f}")
    print(f"{'Diffusion weighting b-value:':<40} {settings.diffusion_weighting_s_m2:.3e} s/m^2")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    assert settings.gradient_duration_s <= 0.005, (
        "Gradient duration exceeds the recommended 5 ms hardware limit."
    )
    assert settings.diffusion_time_s >= settings.gradient_duration_s / 3.0, (
        "Diffusion time must be greater than gradient_duration / 3."
    )

    print("\nAll sanity checks passed. Parameters are physically valid.")


if __name__ == "__main__":
    main()
