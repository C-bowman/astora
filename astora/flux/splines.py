from abc import ABC, abstractmethod
from numpy import ndarray, isfinite, isclose, diff, eye, asarray, concatenate, linspace, ones, zeros
from scipy.interpolate import BSpline, CubicSpline
from scipy.sparse import diags_array, sparray


class ProfileModel(ABC):
    n_knots: int

    @abstractmethod
    def predictions(self, knot_values: ndarray, u: ndarray) -> ndarray:
        pass

    @abstractmethod
    def predictions_and_jacobians(
        self, knot_values: ndarray, u: ndarray
    ) -> tuple[ndarray, dict[str, ndarray | sparray]]:
        pass

    @abstractmethod
    def derivative(self, knot_values: ndarray, u: ndarray) -> ndarray:
        pass

    @abstractmethod
    def derivative_and_jacobians(
        self, knot_values: ndarray, u: ndarray
    ) -> tuple[ndarray, dict[str, ndarray | sparray]]:
        pass


class BSplineProfile(ProfileModel):
    """Open-uniform B-spline profile on 0 <= u <= 1.

    The profile parameters are B-spline coefficients rather than values that
    the profile interpolates at specified locations. This gives each parameter
    compact support. Repeated endpoint knots clamp the basis to the domain.

    :param n_basis: Number of B-spline basis functions and profile parameters.
    :param degree: Polynomial degree of the basis. Must be at least two because
        :meth:`derivative_and_jacobians` evaluates the second derivative.
    """

    def __init__(self, n_basis: int, degree: int = 3):
        if not isinstance(n_basis, int) or isinstance(n_basis, bool):
            raise TypeError("n_basis must be an integer.")
        if not isinstance(degree, int) or isinstance(degree, bool):
            raise TypeError("degree must be an integer.")
        if degree < 2:
            raise ValueError("degree must be at least two.")
        if n_basis < degree + 1:
            raise ValueError("n_basis must be at least degree + one.")

        n_internal_knots = n_basis - degree - 1
        internal_knots = linspace(0.0, 1.0, n_internal_knots + 2)[1:-1]
        self.knots = concatenate(
            (zeros(degree + 1), internal_knots, ones(degree + 1))
        )
        self.degree = degree
        self.n_basis = n_basis
        self.n_knots = n_basis

        self._basis_spline = BSpline(
            self.knots,
            eye(n_basis),
            degree,
            axis=0,
            extrapolate=False,
        )
        self._first_derivative = self._basis_spline.derivative(1)
        self._second_derivative = self._basis_spline.derivative(2)

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
        derivative = self._first_derivative(u) @ knot_values

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
        return self._first_derivative(u) @ knot_values

    def derivative_and_jacobians(
        self,
        knot_values: ndarray,
        u: ndarray,
    ) -> tuple[ndarray, dict[str, ndarray | sparray]]:
        first_derivative_basis = self._first_derivative(u)
        derivative = first_derivative_basis @ knot_values
        second_derivative = self._second_derivative(u) @ knot_values

        jacobians = {
            "knot_values": first_derivative_basis,
            "u": diags_array(second_derivative),
        }
        return derivative, jacobians


class CubicSplineProfile(ProfileModel):
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
