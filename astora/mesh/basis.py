from numpy import array, sqrt, ones, ndarray
from tokamesh.construction import refine_mesh
from tokamesh import TriangularMesh
from astora.diagnostics.magnetics import psi_from_Jtor


def hexacone_basis_points(refinement_level=3):
    # set up a unit hexagon as a mesh
    a = 0.5*sqrt(3)
    points = [(0., 0.), (0., 1.), (a, 0.5), (a, -0.5), (0., -1), (-a, -0.5), (-a, 0.5)]
    triangles = array([[0, i, i % 6 + 1] for i in range(1, 7)])
    R, z = [array([p[i] for p in points]) for i in [0, 1]]
    mesh = TriangularMesh(R=R.copy(), z=z.copy(), triangles=triangles.copy())
    # iteratively refine the mesh to divide the basis function
    # area into many identical triangles
    for _ in range(refinement_level):
        R, z, triangles = refine_mesh(
            R, z, triangles, refinement_bools=ones(triangles.shape[0], dtype=bool)
        )

    # get the centres of the sub-triangles
    R_centre = R[triangles].mean(axis=1)
    z_centre = z[triangles].mean(axis=1)
    area = 3 * sqrt(3) / (2 * R_centre.size)
    # find the basis function weights at the sub-triangle centres
    weights = mesh.build_interpolator_matrix(R=R_centre, z=z_centre)[:, 0]
    return R_centre, z_centre, weights, area


class HexaconeBasis:
    def __init__(self, resolution: float, refinement_level=3):
        self.R_fil, self.z_fil, self.weights, _ = hexacone_basis_points(refinement_level)
        self.R_fil *= resolution
        self.z_fil *= resolution
        self.n_filaments = self.R_fil.size
        self.total_current = 0.5 * sqrt(3) * resolution**2

    def psi_prediction(self, R0: float, z0: float, R: ndarray, z: ndarray) -> ndarray:
        return psi_from_Jtor(
            R0=(self.R_fil + R0)[None, :],
            z0=(self.z_fil + z0)[None, :],
            R=R[:, None],
            z=z[:, None],
        ).sum(axis=1) / self.n_filaments

    def Br_prediction(self, R0: float, z0: float, R: ndarray, z: ndarray, eps=1e-4) -> ndarray:
        f1 = self.psi_prediction(R0, z0, R, z - eps)
        f2 = self.psi_prediction(R0, z0, R, z + eps)
        return (f1 - f2) / (2 * eps * R)

    def Bz_prediction(self, R0: float, z0: float, R: ndarray, z: ndarray, eps=1e-4) -> ndarray:
        f1 = self.psi_prediction(R0, z0, R - eps, z)
        f2 = self.psi_prediction(R0, z0, R + eps, z)
        return (f2 - f1) / (2 * eps * R)
