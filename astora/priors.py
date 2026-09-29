from midas.priors import BasePrior
from midas.parameters import Parameters, Fields
from numpy import ndarray, exp, zeros


class ZeroCurrentBoundary(BasePrior):
    def __init__(
        self, name: str,
        standard_deviation: float,
        boundary_indices: ndarray,
        n_vertices: int,
    ):
        self.name = name
        self.parameters = Parameters(("ln_J", n_vertices))
        self.fields = Fields()

        self.boundary_indices = boundary_indices
        self.sigma = standard_deviation
        self.weight = 1.0 / (standard_deviation**2)

    def probability(self, ln_J: ndarray) -> float:
        boundary_J = exp(ln_J[self.boundary_indices])
        return -0.5 * self.weight * (boundary_J**2).sum()

    def gradients(self, ln_J: ndarray) -> dict[str, ndarray]:
        boundary_J = exp(ln_J[self.boundary_indices])
        grad = -self.weight * boundary_J**2
        gradients = zeros(ln_J.shape)
        gradients[self.boundary_indices] = grad
        return {"ln_J": gradients}