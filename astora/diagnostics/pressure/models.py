from numpy import exp, ndarray, eye, linspace

from astora.flux.transforms import FluxTransform
from astora.diagnostics.magnetics.coils import CoilSet
from astora.mesh.basis import BasisFunction
from midas.parameters import Fields, Parameters, ParameterVector, FieldRequest
from midas.models import DiagnosticModel, FieldModel
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


class PressureFluxProfile(FieldModel):
    def __init__(
        self,
        basis: BasisFunction,
        coil_set: CoilSet,
        flux_transform: FluxTransform,
    ):
        self.name = "pressure"
        self.basis = basis
        self.coils = coil_set
        self.flux_transform = flux_transform
        self.transform_name = self.flux_transform.transform_parameters.name

        self.matrix_cache = {}

        n_profile_knots = 10
        profile_knots = linspace(0.0, 1.0, n_profile_knots)
        self.profile_spline = CubicSplineProfile(profile_knots)

        self.parameters = Parameters(
            ParameterVector(name="ln_J", size=self.basis.n_basis),
            ParameterVector(name="coil_currents", size=self.coils.n_coils),
            ParameterVector(name="pressure_spline_values", size=n_profile_knots),
            self.flux_transform.transform_parameters,
        )

        self.n_params = sum(pv.size for pv in self.parameters)

    def get_psi_matrices(self, field_request: FieldRequest) -> tuple[ndarray, ndarray]:
        if field_request in self.matrix_cache:
            return self.matrix_cache[field_request]

        coils_psi_matrix = self.coils.get_psi_matrix(**field_request.coordinates)
        basis_psi_matrix = self.basis.get_psi_matrix(**field_request.coordinates)

        self.matrix_cache[field_request] = (coils_psi_matrix, basis_psi_matrix)
        return coils_psi_matrix, basis_psi_matrix

    def get_values(
        self,
        parameters: dict[str, ndarray],
        field_request: FieldRequest,
    ) -> ndarray:
        
        basis_J = exp(parameters["ln_J"])
        coils_psi_matrix, basis_psi_matrix = self.get_psi_matrices(field_request)
        psi = basis_psi_matrix @ basis_J + coils_psi_matrix @ parameters["coil_currents"]
        u = self.flux_transform.transform(
            psi=psi,
            parameters=parameters[self.transform_name],
        )

        return self.profile_spline.predictions(
            knot_values=parameters["pressure_spline_values"],
            u=u
        )
    
    def get_values_and_jacobian(
        self,
        parameters: dict[str, ndarray],
        field_request: FieldRequest,
    ) -> tuple[ndarray, dict[str, ndarray]]:
        
        basis_J = exp(parameters["ln_J"])
        coils_psi_matrix, basis_psi_matrix = self.get_psi_matrices(field_request)
        psi = basis_psi_matrix @ basis_J + coils_psi_matrix @ parameters["coil_currents"]
        u, flux_jacobians = self.flux_transform.transform_and_jacobians(
            psi=psi,
            parameters=parameters[self.transform_name],
        )

        values, spline_jacobians = self.profile_spline.predictions_and_jacobians(
            knot_values=parameters["pressure_spline_values"],
            u=u
        )

        pressure_wrt_psi = spline_jacobians["u"] @ flux_jacobians["psi"]
        psi_wrt_ln_J = basis_psi_matrix * basis_J[None, :]

        jacobians = {
            "ln_J": pressure_wrt_psi @ psi_wrt_ln_J,
            "coil_currents": pressure_wrt_psi @ coils_psi_matrix,
            "pressure_spline_values": spline_jacobians["knot_values"],
            self.transform_name: spline_jacobians["u"] @ flux_jacobians["parameters"],
        }

        return values, jacobians

