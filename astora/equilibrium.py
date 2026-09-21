from numpy import arange, array, ndarray, repeat, tile, exp
from scipy.constants import mu_0
from scipy.sparse import csr_array

from astora.diagnostics.magnetics.coils import CoilSet
from astora.mesh.basis import BasisFunction
from midas import Parameters, ParameterVector, Fields, FieldRequest
from midas.priors import BasePrior


def build_finite_difference_operators(
    n_points: int,
    dR: float,
    dz: float,
) -> tuple[csr_array, csr_array, csr_array]:
    """Construct central-difference operators for point-major samples.

    Each point must contribute five consecutive values ordered as
    ``[centre, +dR, -dR, +dz, -dz]``. The returned centre-selection and
    derivative operators all have shape ``(n_points, 5 * n_points)``.
    """
    if n_points < 1:
        raise ValueError("n_points must be positive.")
    if dR <= 0.0 or dz <= 0.0:
        raise ValueError("dR and dz must be positive.")

    rows = repeat(arange(n_points), 2)
    point_offsets = 5 * arange(n_points)
    shape = (n_points, 5 * n_points)

    R_columns = (point_offsets[:, None] + array([1, 2])).ravel()
    z_columns = (point_offsets[:, None] + array([3, 4])).ravel()
    R_weights = tile(array([0.5 / dR, -0.5 / dR]), n_points)
    z_weights = tile(array([0.5 / dz, -0.5 / dz]), n_points)

    centers = csr_array(
        (array([1.0] * n_points), (arange(n_points), point_offsets)),
        shape=shape,
    )
    d_dR = csr_array((R_weights, (rows, R_columns)), shape=shape)
    d_dz = csr_array((z_weights, (rows, z_columns)), shape=shape)
    return centers, d_dR, d_dz


class ForceBalancePrior(BasePrior):
    def __init__(
        self,
        name: str,
        coordinates: dict[str, ndarray],
        basis: BasisFunction,
        coil_set: CoilSet,
        sigma: ndarray
    ):
        self.name = name
        self.coordinates = coordinates
        self.R, self.z = coordinates["R"], coordinates["z"]
        self.dR, self.dz = 1e-3, 1e-3
        self.basis = basis
        self.coil_set = coil_set
        self.sigma = sigma
        self.weight = 1.0 / sigma**2

        R_shifts = array([0.0, self.dR, -self.dR, 0.0, 0.0])
        z_shifts = array([0.0, 0.0, 0.0, self.dz, -self.dz])

        self.R_grid = (self.R[:, None] + R_shifts[None, :]).flatten()
        self.z_grid = (self.z[:, None] + z_shifts[None, :]).flatten()
        self.centers, self.d_dR, self.d_dz = build_finite_difference_operators(
            n_points=self.R.size,
            dR=self.dR,
            dz=self.dz,
        )

        self.coil_Br = self.coil_set.get_Br_matrix(R=self.R, z=self.z)
        self.coil_Bz = self.coil_set.get_Bz_matrix(R=self.R, z=self.z)
        self.plasma_Br = self.basis.get_Br_matrix(R=self.R, z=self.z)
        self.plasma_Bz = self.basis.get_Bz_matrix(R=self.R, z=self.z)
        self.plasma_J = csr_array(
            self.basis.get_interpolator_matrix(R=self.R, z=self.z)
        )

        self.parameters = Parameters(
            ParameterVector(name="ln_J", size=self.basis.n_basis),
            ParameterVector(name="coil_currents", size=self.coil_set.n_coils),
        )

        field_coordinates = {"R": self.R_grid, "z": self.z_grid}
        self.fields = Fields(
            FieldRequest(name="pressure", coordinates=field_coordinates),
            FieldRequest(name="F", coordinates=field_coordinates),
        )

    def probability(
        self,
        ln_J: ndarray,
        coil_currents: ndarray, 
        pressure: ndarray,
        F: ndarray,
    ):
        inverse_mu0_R = 1.0 / (mu_0 * self.R)

        dP_dR = self.d_dR @ pressure
        dP_dz = self.d_dz @ pressure

        J_R = -inverse_mu0_R * (self.d_dz @ F)
        J_z = inverse_mu0_R * (self.d_dR @ F)
        basis_J = exp(ln_J)
        J_phi = self.plasma_J @ basis_J

        B_R = self.coil_Br @ coil_currents + self.plasma_Br @ basis_J
        B_z = self.coil_Bz @ coil_currents + self.plasma_Bz @ basis_J
        B_phi = (self.centers @ F) / self.R

        force_balance_R = J_phi * B_z - J_z * B_phi - dP_dR
        force_balance_z = J_R * B_phi - J_phi * B_R - dP_dz
        force_balance_phi = J_z * B_R - J_R * B_z

        force_sqr = force_balance_R**2 + force_balance_z**2 + force_balance_phi**2
        return -0.5 * (self.weight * force_sqr).sum()

    def gradients(
        self,
        ln_J: ndarray,
        coil_currents: ndarray, 
        pressure: ndarray,
        F: ndarray,
    ) -> dict[str, ndarray]:
        inverse_mu0_R = 1.0 / (mu_0 * self.R)

        dP_dR = self.d_dR @ pressure
        dP_dz = self.d_dz @ pressure

        J_R = -inverse_mu0_R * (self.d_dz @ F)
        J_z = inverse_mu0_R * (self.d_dR @ F)
        basis_J = exp(ln_J)
        J_phi = self.plasma_J @ basis_J

        B_R = self.coil_Br @ coil_currents + self.plasma_Br @ basis_J
        B_z = self.coil_Bz @ coil_currents + self.plasma_Bz @ basis_J
        B_phi = (self.centers @ F) / self.R

        force_balance_R = J_phi * B_z - J_z * B_phi - dP_dR
        force_balance_z = J_R * B_phi - J_phi * B_R - dP_dz
        force_balance_phi = J_z * B_R - J_R * B_z

        force_R_gradient = -self.weight * force_balance_R
        force_z_gradient = -self.weight * force_balance_z
        force_phi_gradient = -self.weight * force_balance_phi

        J_phi_gradient = (
            force_R_gradient * B_z
            - force_z_gradient * B_R
        )
        J_R_gradient = (
            force_z_gradient * B_phi
            - force_phi_gradient * B_z
        )
        J_z_gradient = (
            -force_R_gradient * B_phi
            + force_phi_gradient * B_R
        )
        B_R_gradient = (
            -force_z_gradient * J_phi
            + force_phi_gradient * J_z
        )
        B_z_gradient = (
            force_R_gradient * J_phi
            - force_phi_gradient * J_R
        )
        B_phi_gradient = (
            -force_R_gradient * J_z
            + force_z_gradient * J_R
        )

        basis_J_gradient = (
            self.plasma_J.T @ J_phi_gradient
            + self.plasma_Br.T @ B_R_gradient
            + self.plasma_Bz.T @ B_z_gradient
        )

        return {
            "ln_J": basis_J * basis_J_gradient,
            "coil_currents": (
                self.coil_Br.T @ B_R_gradient
                + self.coil_Bz.T @ B_z_gradient
            ),
            "pressure": (
                self.d_dR.T @ (-force_R_gradient)
                + self.d_dz.T @ (-force_z_gradient)
            ),
            "F": (
                self.d_dz.T @ (-inverse_mu0_R * J_R_gradient)
                + self.d_dR.T @ (inverse_mu0_R * J_z_gradient)
                + self.centers.T @ (B_phi_gradient / self.R)
            ),
        }