from numpy import ndarray, isfinite, isclose, diff, eye, asarray
from scipy.interpolate import CubicSpline
from scipy.sparse import diags_array, sparray


class CubicSplineProfile:
    """Natural cubic-spline profile on 0 <= u <= 1."""

    def __init__(self, knot_positions: ndarray):
        knots = asarray(knot_positions, dtype=float)

        if knots.ndim != 1:
            raise ValueError("knot_positions must be one-dimensional.")
        if knots.size < 2:
            raise ValueError("At least two knot positions are required.")
        if not (isfinite(knots)).all():
            raise ValueError("knot_positions must be finite.")
        if (diff(knots) <= 0.0).any():
            raise ValueError("knot_positions must be strictly increasing.")
        if not isclose(knots[0], 0.0) or not isclose(knots[-1], 1.0):
            raise ValueError("knot_positions must include 0 and 1.")

        self.knot_positions = knots
        self.n_knots = knots.size

        self._basis_spline = CubicSpline(
            knots,
            eye(self.n_knots),
            axis=0,
            bc_type="natural",
            extrapolate=False,
        )

    def predictions(
        self,
        knot_values: ndarray,
        u: ndarray,
    ) -> ndarray:
        return self._basis_spline(u) @ knot_values

    def predictions_and_jacobians(
        self,
        knot_values: ndarray,
        u: ndarray,
    ) -> tuple[ndarray, dict[str, ndarray | sparray]]:

        basis = self._basis_spline(u)
        predictions = basis @ knot_values
        derivative = self._basis_spline(u, 1) @ knot_values

        jacobians = {
            "knot_values": basis,
            "u": diags_array(derivative),
        }
        return predictions, jacobians

    def derivative(
        self,
        knot_values: ndarray,
        u: ndarray,
    ) -> ndarray:
        """Evaluate the profile derivative with respect to u."""
        return self._basis_spline(u, 1) @ knot_values

    def derivative_and_jacobians(
        self,
        knot_values: ndarray,
        u: ndarray,
    ) -> tuple[ndarray, dict[str, ndarray | sparray]]:
        """Evaluate dp/du and its Jacobians."""

        first_derivative_basis = self._basis_spline(u, 1)
        derivative = first_derivative_basis @ knot_values
        second_derivative = self._basis_spline(u, 2) @ knot_values

        jacobians = {
            "knot_values": first_derivative_basis,
            "u": diags_array(second_derivative),
        }
        return derivative, jacobians
