from dataclasses import dataclass
from numpy import ndarray


@dataclass
class TimeSlice:
    measurements: ndarray
    uncertainties: ndarray


@dataclass
class ShotData:
    measurements: ndarray
    uncertainties: ndarray
    times: ndarray

    def __post_init__(self):
        assert self.times.ndim == 1
        assert self.measurements.ndim == 2
        assert self.uncertainties.ndim == 2
        assert self.measurements.shape[0] == self.times.size
        assert self.uncertainties.shape[0] == self.times.size

    def get_slice(self, time_index: int):
        return TimeSlice(
            measurements=self.measurements[time_index, :],
            uncertainties=self.uncertainties[time_index, :],
        )


@dataclass
class FluxloopSpecs:
    R: ndarray
    z: ndarray
    name: ndarray


@dataclass
class FieldSensorSpecs(FluxloopSpecs):
    poloidal_angle: ndarray


@dataclass
class CoilCurrentData:
    measurements: ndarray
    errors: ndarray
    times: ndarray
    names: list[str]
