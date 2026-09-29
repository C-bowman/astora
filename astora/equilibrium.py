from numpy import arange, array, ndarray, repeat, tile, exp, asarray, full, isfinite
from scipy.constants import mu_0
from scipy.sparse import csr_array, sparray

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
        sigma: ndarray,
        delta: float = 1e-3,
    ):
        self.name = name
        self.coordinates = coordinates
        self.R, self.z = coordinates["R"], coordinates["z"]
        self.dR, self.dz = delta, delta
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

        self.inv_mu0_R = 1.0 / (mu_0 * self.R)

    def force_vectors(
        self,
        ln_J: ndarray,
        coil_currents: ndarray,
        pressure: ndarray,
        F: ndarray,
    ):
        dP_dR = self.d_dR @ pressure
        dP_dz = self.d_dz @ pressure

        J_R = -self.inv_mu0_R * (self.d_dz @ F)
        J_z = self.inv_mu0_R * (self.d_dR @ F)
        basis_J = exp(ln_J)
        J_phi = self.plasma_J @ basis_J

        B_R = self.coil_Br @ coil_currents + self.plasma_Br @ basis_J
        B_z = self.coil_Bz @ coil_currents + self.plasma_Bz @ basis_J
        B_phi = (self.centers @ F) / self.R

        return asarray([
            J_phi * B_z - J_z * B_phi - dP_dR,
            J_R * B_phi - J_phi * B_R - dP_dz,
            J_z * B_R - J_R * B_z
        ])

    def probability(
        self,
        ln_J: ndarray,
        coil_currents: ndarray,
        pressure: ndarray,
        F: ndarray,
    ):

        dP_dR = self.d_dR @ pressure
        dP_dz = self.d_dz @ pressure

        J_R = -self.inv_mu0_R * (self.d_dz @ F)
        J_z = self.inv_mu0_R * (self.d_dR @ F)
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

        dP_dR = self.d_dR @ pressure
        dP_dz = self.d_dz @ pressure

        J_R = -self.inv_mu0_R * (self.d_dz @ F)
        J_z = self.inv_mu0_R * (self.d_dR @ F)
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
                self.d_dz.T @ (-self.inv_mu0_R * J_R_gradient)
                + self.d_dR.T @ (self.inv_mu0_R * J_z_gradient)
                + self.centers.T @ (B_phi_gradient / self.R)
            ),
        }


class EquilibriumCurrentPrior(BasePrior):
    """Constrain parameterized current density to Grad-Shafranov predictions."""

    def __init__(
        self,
        name: str,
        basis: BasisFunction,
        coil_set: CoilSet,
        sigma: float,
    ):
        self.name = name
        self.R, self.z = basis.R_basis, basis.z_basis
        self.basis = basis
        self.coil_set = coil_set
        self.sigma = sigma
        self.weight = 1.0 / sigma**2

        self.parameters = Parameters(
            ParameterVector(name="ln_J", size=self.basis.n_basis),
        )

        field_coordinates = {"R": self.R, "z": self.z}
        self.fields = Fields(
            FieldRequest(name="p_prime", coordinates=field_coordinates),
            FieldRequest(name="FF_prime", coordinates=field_coordinates),
        )

        self.inv_mu0_R = 1.0 / (mu_0 * self.R)

    def equilibrium_currents(self, p_prime: ndarray, FF_prime: ndarray):
        return self.R * p_prime + FF_prime * self.inv_mu0_R

    def probability(
        self,
        ln_J: ndarray,
        p_prime: ndarray,
        FF_prime: ndarray
    ) -> float:
        dJ = exp(ln_J) - self.equilibrium_currents(p_prime, FF_prime)
        return -0.5 * self.weight * (dJ**2).sum()

    def gradients(
        self,
        ln_J: ndarray,
        p_prime: ndarray,
        FF_prime: ndarray
    ) -> dict[str, ndarray]:
        J = exp(ln_J)
        dJ = J - self.equilibrium_currents(p_prime, FF_prime)
        dJ_gradient = -self.weight * dJ

        return {
            "ln_J": J * dJ_gradient,
            "p_prime": -self.R * dJ_gradient,
            "FF_prime": -self.inv_mu0_R * dJ_gradient,
        }



def equilateral_mesh_integrator(
    triangles: ndarray,
    edge_length: float
) -> sparray:
    """Build exact per-triangle integrators for a piecewise-linear field.

    The returned sparse array has shape ``(n_triangles, n_vertices)``.
    Multiplying it by field values at the mesh vertices returns the integral
    over each equilateral triangle separately.
    """
    triangles = asarray(triangles)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (n_triangles, 3).")
    if triangles.shape[0] == 0:
        raise ValueError("triangles must contain at least one triangle.")
    if triangles.dtype.kind not in "iu":
        raise TypeError("triangles must contain integer vertex indices.")
    if (triangles < 0).any():
        raise ValueError("triangle vertex indices must be non-negative.")
    if not isfinite(edge_length) or edge_length <= 0.0:
        raise ValueError("edge_length must be positive and finite.")

    n_vertices = int(triangles.max()) + 1
    triangle_area = 0.25 * 3**0.5 * edge_length**2
    vertex_weight = triangle_area / 3.0
    columns = triangles.ravel()
    rows = repeat(arange(triangles.shape[0]), 3)
    weights = full(columns.size, vertex_weight, dtype=float)

    return csr_array(
        (weights, (rows, columns)),
        shape=(triangles.shape[0], n_vertices),
    )