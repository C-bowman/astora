from numpy import exp, ndarray, eye, linspace, diag

from astora.flux.transforms import FluxTransform
from astora.diagnostics.magnetics.coils import CoilSet
from astora.mesh.basis import BasisFunction
from midas.parameters import Fields, Parameters, ParameterVector, FieldRequest
from midas.models import DiagnosticModel
from astora.flux.splines import CubicSplineProfile


class PressurePassthrough(DiagnosticModel):
    def __init__(self, measurement_coordinates: tuple[ndarray, ndarray],):
        self.R, self.z = measurement_coordinates
        self.jacobian = {"pressure": eye(self.R.size)}

        self.parameters = Parameters()
        self.fields = Fields(
            FieldRequest(name="pressure", coordinates={"R": self.R, "z": self.z})
        )

    def predictions(self, pressure) -> ndarray:
        return pressure

    def predictions_and_jacobians(self, pressure) -> tuple[ndarray, dict[str, ndarray]]:
        return pressure, self.jacobian


class PressureModel(DiagnosticModel):
    def __init__(
        self,
        measurement_coordinates: tuple[ndarray, ndarray],
        basis: BasisFunction,
        coil_set: CoilSet,
        flux_transform: FluxTransform,
    ):
        
        self.basis = basis
        self.coils = coil_set
        self.R, self.z = measurement_coordinates
        self.flux_transform = flux_transform
        self.transform_name = self.flux_transform.transform_parameters.name

        self.coils_psi_matrix = self.coils.get_psi_matrix(R=self.R, z=self.z)
        self.basis_psi_matrix = self.basis.get_psi_matrix(R=self.R, z=self.z)

        n_profile_knots = 10
        profile_knots = linspace(0.0, 1.0, n_profile_knots)
        self.profile_spline = CubicSplineProfile(profile_knots)

        self.parameters = Parameters(
            ParameterVector(name="ln_J", size=self.basis.n_basis),
            ParameterVector(name="coil_currents", size=self.coils.n_coils),
            ParameterVector(name="pressure_spline_values", size=n_profile_knots),
            self.flux_transform.transform_parameters,
        )

        self.fields = Fields()


    def predictions(
        self,
        ln_J: ndarray,
        coil_currents: ndarray,
        pressure_spline_values: ndarray,
        **transform_parameter_values: ndarray,
    ) -> ndarray:
        basis_J = exp(ln_J)
        psi = self.basis_psi_matrix @ basis_J + self.coils_psi_matrix @ coil_currents
        u = self.flux_transform.transform(
            psi=psi,
            parameters=transform_parameter_values[self.transform_name],
        )

        return self.profile_spline.predictions(
            knot_values=pressure_spline_values,
            u=u
        )
    
    def predictions_and_jacobians(
        self,
        ln_J: ndarray,
        coil_currents: ndarray,
        pressure_spline_values: ndarray,
        **transform_parameter_values: ndarray,
    ) -> tuple[ndarray, dict[str, ndarray]]:
        
        basis_J = exp(ln_J)
        psi = self.basis_psi_matrix @ basis_J + self.coils_psi_matrix @ coil_currents
        u, flux_jacobians = self.flux_transform.transform_and_jacobians(
            psi=psi,
            parameters=transform_parameter_values[self.transform_name],
        )

        predictions, spline_jacobians = self.profile_spline.predictions_and_jacobians(
            knot_values=pressure_spline_values,
            u=u
        )

        pressure_wrt_psi = spline_jacobians["u"] @ flux_jacobians["psi"]
        psi_wrt_ln_J = self.basis_psi_matrix * basis_J[None, :]

        jacobians = {
            "ln_J": pressure_wrt_psi @ psi_wrt_ln_J,
            "coil_currents": pressure_wrt_psi @ self.coils_psi_matrix,
            "pressure_spline_values": spline_jacobians["knot_values"],
            self.transform_name: spline_jacobians["u"] @ flux_jacobians["parameters"],
        }

        return predictions, jacobians


class RadialPressureGradient(DiagnosticModel):
    def __init__(
        self,
        measurement_coordinates: tuple[ndarray, ndarray],
        basis: BasisFunction,
        coil_set: CoilSet,
    ):

        self.basis = basis
        self.coils = coil_set
        self.R, self.z = measurement_coordinates

        self.coils_dpsi_dR = (
            self.R[:, None] * self.coils.get_Bz_matrix(R=self.R, z=self.z)
        )
        self.basis_dpsi_dR = (
            self.R[:, None] * self.basis.get_Bz_matrix(R=self.R, z=self.z)
        )

        self.parameters = Parameters(
            ParameterVector(name="ln_J", size=self.basis.n_basis),
            ParameterVector(name="coil_currents", size=self.coils.n_coils),
        )

        self.fields = Fields(
            FieldRequest("p_prime", coordinates={"R": self.R, "z": self.z})
        )

    def predictions(self, ln_J: ndarray, coil_currents: ndarray, p_prime: ndarray):
        basis_J = exp(ln_J)
        dpsi_dR = (
            self.basis_dpsi_dR @ basis_J
            + self.coils_dpsi_dR @ coil_currents
        )
        return p_prime * dpsi_dR

    def predictions_and_jacobians(
            self, ln_J: ndarray, coil_currents: ndarray, p_prime: ndarray
    ) -> tuple[ndarray, dict[str, ndarray]]:
        basis_J = exp(ln_J)
        dpsi_dR = (
            self.basis_dpsi_dR @ basis_J
            + self.coils_dpsi_dR @ coil_currents
        )
        predictions = p_prime * dpsi_dR

        jacobians = {
            "ln_J": p_prime[:, None] * self.basis_dpsi_dR * basis_J[None, :],
            "coil_currents": p_prime[:, None] * self.coils_dpsi_dR,
            "p_prime": diag(dpsi_dR),
        }

        return predictions, jacobians

