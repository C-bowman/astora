from dataclasses import dataclass
from numpy import ndarray


@dataclass
class FluxloopData:
    R: ndarray
    z: ndarray
    name: ndarray
    measurements: ndarray
    errors: ndarray
    computed: ndarray


@dataclass
class FieldSensorData(FluxloopData):
    poloidal_angle: ndarray


@dataclass
class CurrentData:
    measurements: ndarray
    errors: ndarray
    computed: ndarray
