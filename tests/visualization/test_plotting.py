import numpy as np

from nmrkit.core.data import DimensionInfo, LinearGenerator, NMRData
from nmrkit.visualization.plotting import _plot_2d


class FakePlot:
    def __init__(self):
        self.contour_args = None
        self.xlim_args = None
        self.ylim_args = None

    def figure(self, *args, **kwargs):
        pass

    def contour(self, x, y, signal, *args, **kwargs):
        self.contour_args = (x.copy(), y.copy(), signal.copy())

    def xlabel(self, *args, **kwargs):
        pass

    def ylabel(self, *args, **kwargs):
        pass

    def xlim(self, *args):
        self.xlim_args = args

    def ylim(self, *args):
        self.ylim_args = args

    def tight_layout(self):
        pass

    def show(self):
        pass


def test_plot_2d_uses_direct_dimension_as_x_axis():
    data_array = np.arange(12, dtype=float).reshape(3, 4)
    nmr_data = NMRData(
        data=data_array,
        dimensions=[
            DimensionInfo(
                size=3,
                domain_type="frequency",
                unit="Hz",
                observation_frequency=1.0,
                axis_generator=LinearGenerator(start=10.0, step=1.0),
            ),
            DimensionInfo(
                size=4,
                domain_type="frequency",
                unit="Hz",
                observation_frequency=1.0,
                axis_generator=LinearGenerator(start=100.0, step=10.0),
            ),
        ],
    )
    fake = FakePlot()

    _plot_2d(nmr_data, plt=fake)

    x, y, signal = fake.contour_args
    np.testing.assert_array_equal(x, np.array([12.0, 11.0, 10.0]))
    np.testing.assert_array_equal(y, np.array([130.0, 120.0, 110.0, 100.0]))
    np.testing.assert_array_equal(signal, np.flip(data_array.T, axis=(0, 1)))
    assert signal.shape == (4, 3)
