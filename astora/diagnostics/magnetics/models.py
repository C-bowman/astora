from numpy import exp, sin, cos, ndarray

from .data import FluxloopSpecs, FieldSensorSpecs
from .coils import CoilSet
from astora.mesh.basis import BasisFunction
from midas.parameters import ParameterVector
from midas.models import DiagnosticModel


class FluxloopModel(DiagnosticModel):
    def __init__(
        self,
        fluxloop_specs: FluxloopSpecs,
        basis: BasisFunction,
        coil_set: CoilSet
    ):
        self.specs = fluxloop_specs
        self.basis = basis
        self.coils = coil_set

        self.coil_matrix = self.coils.get_psi_matrix(R=self.specs.R, z=self.specs.z)
        self.basis_matrix = self.basis.get_psi_matrix(R=self.specs.R, z=self.specs.z)

        self.parameters = [
            ParameterVector(name="ln_J", size=self.basis.n_basis),
            ParameterVector(name="coil_currents", size=self.coils.n_coils)
        ]

        self.field_requests = []

    def predictions(self, ln_J: ndarray, coil_currents: ndarray):
        basis_currents = exp(ln_J)
        return self.basis_matrix @ basis_currents + self.coil_matrix @ coil_currents

    def predictions_and_jacobians(self, ln_J: ndarray, coil_currents: ndarray):
        basis_currents = exp(ln_J)
        predictions = self.basis_matrix @ basis_currents + self.coil_matrix @ coil_currents
        jacobians = {
            "ln_J": self.basis_matrix * basis_currents[None, :],
            "coil_currents": self.coil_matrix
        }
        return predictions, jacobians


class FieldSensorModel(DiagnosticModel):
    def __init__(
        self,
        field_sensor_specs: FieldSensorSpecs,
        basis: BasisFunction,
        coil_set: CoilSet
    ):
        self.specs = field_sensor_specs
        self.basis = basis
        self.coils = coil_set

        M_IR = self.coils.get_Br_matrix(R=self.specs.R, z=self.specs.z)
        M_Iz = self.coils.get_Bz_matrix(R=self.specs.R, z=self.specs.z)
        M_JR = self.basis.get_Br_matrix(R=self.specs.R, z=self.specs.z)
        M_Jz = self.basis.get_Bz_matrix(R=self.specs.R, z=self.specs.z)

        sin_t = sin(self.specs.poloidal_angle)
        cos_t = cos(self.specs.poloidal_angle)
        self.basis_matrix = (cos_t[:, None] * M_JR + sin_t[:, None] * M_Jz)
        self.coil_matrix = (cos_t[:, None] * M_IR + sin_t[:, None] * M_Iz)

        self.parameters = [
            ParameterVector(name="ln_J", size=self.basis.n_basis),
            ParameterVector(name="coil_currents", size=self.coils.n_coils)
        ]

        self.field_requests = []

    def predictions(self, ln_J: ndarray, coil_currents: ndarray):
        basis_J = exp(ln_J)
        return self.basis_matrix @ basis_J + self.coil_matrix @ coil_currents

    def predictions_and_jacobians(self, ln_J: ndarray, coil_currents: ndarray):
        basis_J = exp(ln_J)
        predictions = self.basis_matrix @ basis_J + self.coil_matrix @ coil_currents
        jacobians = {
            "ln_J": self.basis_matrix * basis_J[None, :],
            "coil_currents": self.coil_matrix
        }
        return predictions, jacobians


class PlasmaCurrentModel(DiagnosticModel):
    def __init__(self, basis: BasisFunction):
        self.basis = basis

        self.parameters = [ParameterVector(name="ln_J", size=self.basis.n_basis)]
        self.field_requests = []

    def predictions(self, ln_J: ndarray):
        return exp(ln_J).sum() * self.basis.total_current

    def predictions_and_jacobians(self, ln_J: ndarray) -> tuple[ndarray, dict[str, ndarray]]:
        basis_I = exp(ln_J) * self.basis.total_current
        predictions = basis_I.sum()
        jacobians = {
            "ln_J": basis_I.reshape(1, self.basis.n_basis),
        }
        return predictions, jacobians


class MidplanePressureModel(DiagnosticModel):
    def __init__(
        self,
        pressure_data,
        basis: BasisFunction,
        coil_set: CoilSet
    ):
        self.data = pressure_data
        self.basis = basis
        self.coils = coil_set

        self.coils_Bz_matrix = self.coils.get_Bz_matrix(R=self.data.R, z=self.data.z)
        self.basis_Bz_matrix = self.basis.get_Bz_matrix(R=self.data.R, z=self.data.z)
        self.basis_J_matrix = self.basis.get_interpolator_matrix(R=self.data.R, z=self.data.z)

        self.parameters = [
            ParameterVector(name="ln_J", size=self.basis.n_basis),
            ParameterVector(name="coil_currents", size=self.coils.n_coils)
        ]

        self.field_requests = []

    def predictions(self, ln_J: ndarray, coil_currents: ndarray):
        basis_J = exp(ln_J)
        Bz = self.basis_Bz_matrix @ basis_J + self.coils_Bz_matrix @ coil_currents
        J = self.basis_J_matrix @ basis_J
        grad_p = J * Bz / self.data.R
        predictions = None
        return predictions

    def predictions_and_jacobians(self, ln_J: ndarray, coil_currents: ndarray):
        basis_J = exp(ln_J)
        Bz = self.basis_Bz_matrix @ basis_J + self.coils_Bz_matrix @ coil_currents
        J = self.basis_J_matrix @ basis_J
        grad_p = J * Bz / self.data.R
        predictions = None
        jacobians = {
            "ln_J": None,
            "coil_currents": None
        }
        return predictions, jacobians