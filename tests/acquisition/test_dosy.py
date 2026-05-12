import numpy as np
import pytest

import nmrkit as nk
from nmrkit.acquisition.dosy import (
    DOSYSettings,
    attenuation_factor,
    calculate_dosy_settings,
    diffusion_weighting,
    gamma_for_nucleus,
)
from nmrkit.constants import PROTON_GYROMAGNETIC_RATIO_RAD_S_T


def test_gamma_for_nucleus_uses_shared_constants():
    assert gamma_for_nucleus("1H") == PROTON_GYROMAGNETIC_RATIO_RAD_S_T

    with pytest.raises(ValueError):
        gamma_for_nucleus("2H")


def test_diffusion_weighting_matches_stejskal_tanner_equation():
    gamma = gamma_for_nucleus("1H")
    result = diffusion_weighting(
        gradient_t_m=0.535,
        diffusion_time_s=0.05,
        gradient_duration_s=0.0022,
        gamma_rad_s_t=gamma,
    )
    expected = (gamma * 0.535 * 0.0022) ** 2 * (0.05 - 0.0022 / 3.0)

    assert result == pytest.approx(expected)


def test_attenuation_factor_applies_diffusion_coefficient():
    attenuation = attenuation_factor(
        diffusion_coefficient_m2_s=5.8e-10,
        gradient_t_m=0.535,
        diffusion_time_s=0.05,
        gradient_duration_s=0.0022,
    )

    assert attenuation == pytest.approx(0.058842, rel=1e-4)


def test_calculate_dosy_settings_uses_default_100ms_diffusion_time():
    """For moderate D and G, default 100 ms diffusion time should be used."""
    settings = calculate_dosy_settings(
        diffusion_coefficient_m2_s=5.8e-10,
        max_gradient_t_m=0.535,
        target_attenuation=0.05,
    )

    assert isinstance(settings, DOSYSettings)
    assert settings.diffusion_time_s == pytest.approx(0.100)
    assert settings.gradient_duration_s <= 0.005
    assert settings.achieved_attenuation == pytest.approx(0.05, rel=1e-3)
    assert settings.diffusion_time_ms == pytest.approx(100.0)
    assert settings.gradient_duration_ms == pytest.approx(
        settings.gradient_duration_s * 1000.0
    )


def test_calculate_dosy_settings_adjusts_diffusion_time_when_needed():
    """For small D or large G, gradient duration may exceed 5 ms at 100 ms,
    so diffusion time should be increased."""
    settings = calculate_dosy_settings(
        diffusion_coefficient_m2_s=1.0e-11,
        max_gradient_t_m=1.0,
        target_attenuation=0.05,
    )

    assert settings.gradient_duration_s <= 0.005
    assert settings.achieved_attenuation == pytest.approx(0.05, rel=1e-3)


def test_calculate_dosy_settings_supports_custom_target_attenuation():
    settings = calculate_dosy_settings(
        diffusion_coefficient_m2_s=5.8e-10,
        max_gradient_t_m=0.535,
        target_attenuation=0.10,
    )

    assert settings.target_attenuation == pytest.approx(0.10)
    assert settings.achieved_attenuation == pytest.approx(0.10, rel=1e-3)


def test_calculate_dosy_settings_warns_when_diffusion_time_exceeds_200ms():
    """Very small diffusion coefficients may require diffusion time > 200 ms."""
    with pytest.warns(UserWarning, match="T1"):
        settings = calculate_dosy_settings(
            diffusion_coefficient_m2_s=1.0e-12,
            max_gradient_t_m=0.535,
            target_attenuation=0.05,
        )

    assert settings.diffusion_time_s > 0.200
    assert settings.gradient_duration_s <= 0.005


def test_top_level_exports_acquisition_helpers():
    assert nk.calculate_dosy_settings is calculate_dosy_settings
