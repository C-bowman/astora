from collections.abc import Callable

from numpy import asarray, linspace, exp, ndarray
from midas import FieldRequest, Parameters, ParameterVector
from midas.models import FieldModel

from astora.diagnostics.magnetics.coils import CoilSet
from astora.mesh.basis import BasisFunction
from astora.flux.splines import BSplineProfile, CubicSplineProfile, ProfileModel
from astora.flux.transforms import FluxTransform


class FluxProfile(FieldModel):
    def __init__(
        self,
        field_name: str,
        basis: BasisFunction,
        coil_set: CoilSet,
        flux_transform: FluxTransform,
        n_profile_knots: int = 10,
        profile_model: ProfileModel | None = None,
    ):
        if not isinstance(n_profile_knots, int) or isinstance(n_profile_knots, bool):
            raise TypeError("n_profile_knots must be an integer.")
        if n_profile_knots < 2:
            raise ValueError("n_profile_knots must be at least two.")

        self.name = field_name
        self.spline_name = f"{field_name}_spline_values"
        self.basis = basis
        self.coils = coil_set
        self.flux_transform = flux_transform
        self.transform_name = self.flux_transform.transform_parameters.name

        self.matrix_cache = {}

        if profile_model is None:
            profile_knots = linspace(0.0, 1.0, n_profile_knots)
            self.profile_spline = CubicSplineProfile(profile_knots)
        elif isinstance(profile_model, ProfileModel):
            self.profile_spline = profile_model
        else:
            raise TypeError("profile_model must be a ProfileModel.")

        n_profile_parameters = self.profile_spline.n_knots

        self.parameters = Parameters(
            ParameterVector(name="ln_J", size=self.basis.n_basis),
            ParameterVector(name="coil_currents", size=self.coils.n_coils),
            ParameterVector(name=self.spline_name, size=n_profile_parameters),
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
            knot_values=parameters[self.spline_name],
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
            knot_values=parameters[self.spline_name],
            u=u
        )

        pressure_wrt_psi = spline_jacobians["u"] @ flux_jacobians["psi"]
        psi_wrt_ln_J = basis_psi_matrix * basis_J[None, :]

        jacobians = {
            "ln_J": pressure_wrt_psi @ psi_wrt_ln_J,
            "coil_currents": pressure_wrt_psi @ coils_psi_matrix,
            self.spline_name: spline_jacobians["knot_values"],
            self.transform_name: spline_jacobians["u"] @ flux_jacobians["parameters"],
        }

        return values, jacobians

    def get_values_and_pullback(
        self,
        parameters: dict[str, ndarray],
        field_request: FieldRequest,
    ) -> tuple[ndarray, Callable[[ndarray], dict[str, ndarray]]]:
        basis_J = exp(parameters["ln_J"])
        coils_psi_matrix, basis_psi_matrix = self.get_psi_matrices(field_request)
        psi = (
            basis_psi_matrix @ basis_J
            + coils_psi_matrix @ parameters["coil_currents"]
        )
        u, flux_jacobians = self.flux_transform.transform_and_jacobians(
            psi=psi,
            parameters=parameters[self.transform_name],
        )

        values, spline_jacobians = self.profile_spline.predictions_and_jacobians(
            knot_values=parameters[self.spline_name],
            u=u,
        )

        def pullback(vector: ndarray) -> dict[str, ndarray]:
            u_gradient = asarray(spline_jacobians["u"].T @ vector).ravel()
            psi_gradient = asarray(
                flux_jacobians["psi"].T @ u_gradient
            ).ravel()
            return {
                "ln_J": basis_J * (basis_psi_matrix.T @ psi_gradient),
                "coil_currents": coils_psi_matrix.T @ psi_gradient,
                self.spline_name: spline_jacobians["knot_values"].T @ vector,
                self.transform_name: flux_jacobians["parameters"].T @ u_gradient,
            }

        return values, pullback
