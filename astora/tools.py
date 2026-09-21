from collections.abc import Callable
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from numpy import ndarray
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import root

from tokamesh.construction import Polygon


def dist(u: ndarray, v: ndarray) -> float:
    """Return the Euclidean distance between two points."""
    return float(norm(u - v))

def unit(vector: ndarray) -> ndarray:
    """Return a unit vector in the same direction as ``vector``."""
    magnitude = norm(vector)
    if magnitude == 0.0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / magnitude


class NormalisedPoloidalFlux:
    """Interpolate and trace an axisymmetric poloidal-flux equilibrium."""

    def __init__(
        self, R: ndarray, z: ndarray, psi: ndarray, boundary: Polygon | None = None
    ) -> None:
        self.R = np.asarray(R, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.psi_grid = np.asarray(psi, dtype=float)
        self._validate_grid()

        self.psi_spline = RectBivariateSpline(self.R, self.z, self.psi_grid)
        self.R_min, self.R_max = float(self.R[0]), float(self.R[-1])
        self.z_min, self.z_max = float(self.z[0]), float(self.z[-1])

        self.boundary: Polygon | None = boundary
        self.maxima: list[ndarray] = []
        self.minima: list[ndarray] = []
        self.x_points: list[ndarray] = []
        self.lower_x_point: ndarray | None = None
        self.upper_x_point: ndarray | None = None

        self.normalise_flux()

    def _validate_grid(self) -> None:
        if self.R.ndim != 1 or self.z.ndim != 1:
            raise ValueError("R and z must be one-dimensional.")
        if self.R.size < 4 or self.z.size < 4:
            raise ValueError("R and z must each contain at least four points.")
        if np.any(np.diff(self.R) <= 0.0) or np.any(np.diff(self.z) <= 0.0):
            raise ValueError("R and z must be strictly increasing.")
        expected_shape = (self.R.size, self.z.size)
        if self.psi_grid.shape != expected_shape:
            raise ValueError(
                f"psi must have shape {expected_shape}, got {self.psi_grid.shape}."
            )
        if not all(
            np.all(np.isfinite(values))
            for values in (self.R, self.z, self.psi_grid)
        ):
            raise ValueError("R, z, and psi must contain only finite values.")

    def _in_bounds(self, point: ndarray) -> bool:
        R, z = np.asarray(point)
        return self.R_min < R < self.R_max and self.z_min < z < self.z_max

    def _filter_inside_boundary(self, points: list[ndarray]) -> list[ndarray]:
        if self.boundary is None or not points:
            return points

        coordinates = np.asarray(points)
        inside = self.boundary.is_inside(coordinates[:, 0], coordinates[:, 1])
        return [
            point
            for point, is_inside in zip(points, inside, strict=True)
            if is_inside
        ]

    def __call__(self, coordinates: ndarray, **kwargs: int) -> ndarray:
        return self.psi(coordinates, **kwargs)

    def psi(self, coordinates: ndarray, **kwargs: int) -> ndarray:
        """Evaluate the poloidal flux or one of its spline derivatives."""
        return self.psi_spline.ev(*coordinates, **kwargs).squeeze()

    def grad(self, coordinates: ndarray) -> ndarray:
        """Return the gradient of poloidal flux with respect to ``(R, z)``."""
        return np.asarray(
            [self.psi(coordinates, dx=1), self.psi(coordinates, dy=1)]
        )

    def grad_drn(self, coordinates: ndarray) -> ndarray:
        """Return the unit vector parallel to the flux gradient."""
        return unit(self.grad(coordinates))

    def perp_grad_drn(
        self,
        coordinates: ndarray,
        direction: int = 1,
    ) -> ndarray:
        """Return a unit vector tangent to the local flux surface."""
        if direction not in (-1, 1):
            raise ValueError("direction must be either -1 or 1.")
        gradient = self.grad(coordinates)
        perpendicular = np.asarray([gradient[1], -gradient[0]])
        return direction * unit(perpendicular)

    def nabla(self, coordinates: ndarray) -> float:
        """Return the squared magnitude of the poloidal-flux gradient."""
        gradient = self.grad(coordinates)
        return float(gradient @ gradient)

    def grad_nabla(self, coordinates: ndarray) -> ndarray:
        """Return the gradient of the squared flux-gradient magnitude."""
        return 2.0 * self.hessian(coordinates) @ self.grad(coordinates)

    def hessian(self, coordinates: ndarray) -> ndarray:
        """Return the Hessian of poloidal flux with respect to ``(R, z)``."""
        return np.asarray(
            [
                [self.psi(coordinates, dx=2), self.psi(coordinates, dx=1, dy=1)],
                [self.psi(coordinates, dx=1, dy=1), self.psi(coordinates, dy=2)],
            ]
        )

    def newton_update(self, coordinates: ndarray, target: float) -> ndarray:
        """Return a normal correction that moves a point toward a flux value."""
        gradient = self.grad(coordinates)
        gradient_norm_squared = float(gradient @ gradient)
        if gradient_norm_squared == 0.0:
            raise ValueError("Cannot compute a Newton update at a stationary point.")
        return (target - self.psi(coordinates)) * gradient / gradient_norm_squared

    def follow_gradient(
        self,
        start: ndarray,
        target_psi: float,
        max_step_size: float = 5e-3,
        max_steps: int = 2_000,
    ) -> ndarray:
        """Follow the flux gradient from ``start`` to ``target_psi``."""
        if max_step_size <= 0.0:
            raise ValueError("max_step_size must be positive.")
        if max_steps < 1:
            raise ValueError("max_steps must be positive.")

        point = np.asarray(start, dtype=float).copy()
        initial_delta = target_psi - self.psi(point)
        if np.isclose(initial_delta, 0.0):
            return point

        direction = float(np.sign(initial_delta))
        flux_distance = abs(initial_delta)
        for _ in range(max_steps):
            if direction * (target_psi - self.psi(point)) <= 0.0:
                break
            gradient = self.grad(point)
            magnitude = norm(gradient)
            if magnitude == 0.0:
                raise RuntimeError("Encountered a stationary point before the target flux.")
            step_size = direction * min(
                0.05 * flux_distance / magnitude,
                max_step_size,
            )
            point += step_size * gradient / magnitude
        else:
            raise RuntimeError("Failed to reach the target flux within max_steps.")

        for _ in range(2):
            point += self.newton_update(point, target=target_psi)
        return point

    def follow_surface(
        self,
        start: ndarray,
        distance: float,
        step_size: float = 1e-3,
        max_steps: int = 2_000,
        direction: int = 1,
    ) -> ndarray:
        """Follow a flux surface for a given poloidal distance."""
        if distance < 0.0:
            raise ValueError("distance must be non-negative.")
        if step_size <= 0.0:
            raise ValueError("step_size must be positive.")
        if max_steps < 1:
            raise ValueError("max_steps must be positive.")

        point = np.asarray(start, dtype=float).copy()
        target_psi = self.psi(point)
        travelled = 0.0

        for _ in range(max_steps):
            remaining = distance - travelled
            if remaining < 1e-6:
                return point

            candidate = point + step_size * self.perp_grad_drn(point, direction)
            candidate += self.newton_update(candidate, target=target_psi)
            candidate_distance = dist(point, candidate)

            if candidate_distance <= remaining:
                travelled += candidate_distance
                point = candidate
            else:
                step_size = 0.95 * remaining

        raise RuntimeError("Failed to travel the requested distance within max_steps.")

    def follow_surface_while(
        self,
        start: ndarray,
        condition: Callable[[ndarray], bool],
        step_size: float = 1e-3,
        max_steps: int = 2_000,
        direction: int = 1,
    ) -> tuple[ndarray, float]:
        """Follow a flux surface while ``condition`` accepts each new point."""
        if step_size <= 0.0:
            raise ValueError("step_size must be positive.")
        if max_steps < 1:
            raise ValueError("max_steps must be positive.")

        point = np.asarray(start, dtype=float).copy()
        target_psi = self.psi(point)
        travelled = 0.0

        for _ in range(max_steps):
            candidate = point + step_size * self.perp_grad_drn(point, direction)
            candidate += self.newton_update(candidate, target=target_psi)
            if condition(candidate):
                travelled += dist(point, candidate)
                point = candidate
            else:
                step_size *= 0.5
            if step_size < 1e-6:
                return point, travelled

        raise RuntimeError("Surface tracing did not satisfy the stopping tolerance.")

    def find_stationary_points(
        self,
        R_points: int = 5,
        z_points: int = 25,
    ) -> list[ndarray]:
        """Find and classify stationary points inside the flux grid."""
        if R_points < 1 or z_points < 1:
            raise ValueError("R_points and z_points must be positive.")

        R_starts = np.linspace(self.R_min, self.R_max, R_points + 2)[1:-1]
        z_starts = np.linspace(self.z_min, self.z_max, z_points + 2)[1:-1]
        candidates: list[ndarray] = []

        for R_start, z_start in product(R_starts, z_starts):
            result = root(
                self.grad,
                np.asarray([R_start, z_start]),
                jac=self.hessian,
            )
            if (
                result.success
                and self._in_bounds(result.x)
                and norm(self.grad(result.x)) < 1e-7
            ):
                candidates.append(result.x)

        stationary_points: list[ndarray] = []
        for candidate in candidates:
            if all(dist(candidate, point) >= 1e-4 for point in stationary_points):
                stationary_points.append(candidate)

        self.maxima = []
        self.minima = []
        self.x_points = []
        for point in stationary_points:
            eigenvalues = np.linalg.eigvalsh(self.hessian(point))
            if eigenvalues[0] > 0.0:
                self.minima.append(point)
            elif eigenvalues[1] < 0.0:
                self.maxima.append(point)
            elif eigenvalues[0] < 0.0 < eigenvalues[1]:
                self.x_points.append(point)

        self.maxima.sort(key=self.psi)
        self.minima.sort(key=self.psi)
        self.x_points.sort(key=self.psi)

        if self.boundary is not None:
            self.maxima = self._filter_inside_boundary(self.maxima)
            self.minima = self._filter_inside_boundary(self.minima)
            self.x_points = self._filter_inside_boundary(self.x_points)

        return stationary_points

    def _select_magnetic_axis(self) -> ndarray:
        candidates = [
            *((point, 1.0) for point in self.minima),
            *((point, -1.0) for point in self.maxima),
        ]
        if not candidates:
            raise RuntimeError("No O-point candidate was found in the flux grid.")
        if not self.x_points:
            raise RuntimeError("No X-point candidate was found in the flux grid.")

        x_point_flux = np.asarray([self.psi(point) for point in self.x_points])

        def flux_well_depth(candidate: tuple[ndarray, float]) -> float:
            point, direction = candidate
            separations = direction * (x_point_flux - self.psi(point))
            outward_separations = separations[separations > 0.0]
            if outward_separations.size == 0:
                return -np.inf
            return float(outward_separations.min())

        magnetic_axis, axis_direction = max(candidates, key=flux_well_depth)
        if not np.isfinite(flux_well_depth((magnetic_axis, axis_direction))):
            raise RuntimeError("No O-point could be paired with an outward X-point.")
        return magnetic_axis

    def normalise_flux(self) -> None:
        """Normalize flux to zero on axis and one at the LCFS."""
        self.find_stationary_points()
        self.magnetic_axis = self._select_magnetic_axis()
        self.psi_lcfs = self.find_lcfs_psi(self.magnetic_axis)

        self.x_points.sort(
            key=lambda point: abs(self.psi(point) - self.psi_lcfs)
        )
        self.primary_x_point = self.x_points[0]

        self.normalisation_offset = float(self.psi(self.magnetic_axis))
        flux_span = self.psi_lcfs - self.normalisation_offset
        if np.isclose(flux_span, 0.0):
            raise RuntimeError("Axis and LCFS flux values are indistinguishable.")
        self.normalisation_scaling = 1.0 / flux_span
        self.psi_grid = (
            self.psi_grid - self.normalisation_offset
        ) * self.normalisation_scaling
        self.psi_spline = RectBivariateSpline(self.R, self.z, self.psi_grid)

        self.lower_x_point = None
        self.upper_x_point = None
        if len(self.x_points) > 1 and self.psi(self.x_points[1]) < 1.01:
            self.lower_x_point, self.upper_x_point = sorted(
                self.x_points[:2], key=lambda point: point[1]
            )
        elif self.primary_x_point[1] < self.magnetic_axis[1]:
            self.lower_x_point = self.primary_x_point
        else:
            self.upper_x_point = self.primary_x_point

    def lcfs_directions(self, null: str | None = None) -> list[ndarray]:
        """Return the four separatrix directions at an X-point."""
        if null not in (None, "lower", "upper"):
            raise ValueError("null must be None, 'lower', or 'upper'.")

        x_point = {
            None: self.primary_x_point,
            "lower": self.lower_x_point,
            "upper": self.upper_x_point,
        }[null]
        if x_point is None:
            raise ValueError(f"This equilibrium has no {null} X-point.")

        theta = np.linspace(0.0, 2.0 * np.pi, 4 * 360, endpoint=False)
        radius = 0.01
        lcfs_deviation = (
            1.0
            - self.psi(
                np.asarray(
                    [
                    x_point[0] + radius * np.cos(theta),
                    x_point[1] + radius * np.sin(theta),
                    ]
                )
            )
        ) ** 2
        lcfs_angle = theta[lcfs_deviation.argmin()]
        directions = [
            np.asarray([np.cos(lcfs_angle + angle), np.sin(lcfs_angle + angle)])
            for angle in np.arange(4) * 0.5 * np.pi
        ]

        axis_direction = unit(self.magnetic_axis - x_point)
        outboard_direction = np.asarray([1.0, 0.0])
        directions.sort(key=lambda direction: float(direction @ axis_direction))
        directions.sort(key=lambda direction: float(direction @ outboard_direction))
        return directions

    def plot_equilibrium(
        self,
        flux_range: tuple[float, float] = (0.0, 1.5),
        axis: plt.Axes | None = None,
    ) -> None:
        """Plot normalized flux contours and the optional domain boundary."""

        if axis is None:
            aspect = (self.R_max - self.R_min) / (self.z_max - self.z_min)
            _, axis = plt.subplots(figsize=(8.0 * aspect, 8.0))

        axis.contour(
            self.R,
            self.z,
            self.psi_grid.T,
            levels=np.linspace(*flux_range, 24),
        )
        axis.contour(self.R, self.z, self.psi_grid.T, levels=[1.0], colors=["red"])

        if self.boundary is not None:
            axis.plot(self.boundary.x, self.boundary.y, linewidth=4, color="white")
            axis.plot(self.boundary.x, self.boundary.y, linewidth=2, color="black")
        axis.set(xlim=(self.R_min, self.R_max), ylim=(self.z_min, self.z_max))
        axis.set_aspect("equal")

        if axis is None:
            plt.show()

    def plot_stationary_points(self) -> None:
        """Plot flux contours and all classified stationary points."""
        R_mesh, z_mesh = np.meshgrid(
            np.linspace(self.R_min, self.R_max, 64),
            np.linspace(self.z_min, self.z_max, 128),
        )
        self.find_stationary_points()

        aspect = (self.R_max - self.R_min) / (self.z_max - self.z_min)
        _, axes = plt.subplots(figsize=(8.0 * aspect, 8.0))
        psi_mesh = self.psi(np.asarray([R_mesh, z_mesh]))
        axes.contour(R_mesh, z_mesh, psi_mesh, 64)
        styles = (
            (self.x_points, "x", "red", "X-points"),
            (self.minima, "o", "dodgerblue", "Minima"),
            (self.maxima, "o", "orange", "Maxima"),
        )
        for points, marker, color, label in styles:
            if points:
                coordinates = np.asarray(points)
                axes.scatter(
                    coordinates[:, 0],
                    coordinates[:, 1],
                    marker=marker,
                    color=color,
                    edgecolors="black" if marker == "o" else None,
                    label=label,
                )

        axes.set(xlim=(self.R_min, self.R_max), ylim=(self.z_min, self.z_max))
        axes.set_aspect("equal")
        axes.legend()
        plt.show()

    def get_separatrix(
        self,
        resolution_multiplier: float | None = None,
    ) -> tuple[ndarray, ndarray]:
        """Return coordinates of the longest normalized-flux-one contour."""
        if resolution_multiplier is None:
            R_values = self.R
            z_values = self.z
        else:
            if resolution_multiplier <= 0.0:
                raise ValueError("resolution_multiplier must be positive.")
            R_size = max(4, round(resolution_multiplier * self.R.size))
            z_size = max(4, round(resolution_multiplier * self.z.size))
            R_values = np.linspace(self.R_min, self.R_max, R_size)
            z_values = np.linspace(self.z_min, self.z_max, z_size)

        R_mesh, z_mesh = np.meshgrid(R_values, z_values)
        psi_mesh = self.psi(np.asarray([R_mesh, z_mesh]))
        figure, axes = plt.subplots()
        try:
            contour = axes.contour(R_mesh, z_mesh, psi_mesh, levels=[1.0])
            segments = [
                segment for segment in contour.allsegs[0] if segment.shape[0] > 1
            ]
        finally:
            plt.close(figure)
        if not segments:
            raise RuntimeError("No separatrix contour was found in the flux grid.")

        vertices = max(segments, key=len)
        return vertices[:, 0], vertices[:, 1]

    def find_lcfs_psi(self, magnetic_axis: ndarray) -> float:
        """Find the flux value of the last closed flux surface."""
        axis = np.asarray(magnetic_axis, dtype=float)
        if axis.shape != (2,) or not self._in_bounds(axis):
            raise ValueError("magnetic_axis must be a point inside the flux grid.")

        start = axis.copy()
        displacement = 3e-2
        R_lower = float(axis[0])
        R_upper = R_lower + displacement

        while R_upper < self.R_max:
            start[0] = R_upper
            if not self.closed_check(start, axis):
                break
            R_lower = R_upper
            R_upper += displacement
        else:
            raise RuntimeError("No open flux surface was found before the grid edge.")

        while R_upper - R_lower > 1e-4:
            start[0] = 0.5 * (R_upper + R_lower)
            if self.closed_check(start, axis):
                R_lower = start[0]
            else:
                R_upper = start[0]

        start[0] = 0.5 * (R_upper + R_lower)
        return float(self.psi(start))

    def closed_check(self, start: ndarray, magnetic_axis: ndarray) -> bool:
        """Return whether the flux surface through ``start`` closes around the axis."""
        start_point = np.asarray(start, dtype=float)
        axis = np.asarray(magnetic_axis, dtype=float)
        point = start_point.copy()
        target_psi = self.psi(start_point)

        step_size = 1e-2
        initial_step_size = step_size
        has_left_start = False
        full_turn = False

        for _ in range(2_000):
            candidate = point + step_size * self.perp_grad_drn(point, direction=1)
            candidate += self.newton_update(candidate, target=target_psi)
            distance_from_start = dist(candidate, start_point)
            has_left_start = (
                has_left_start or distance_from_start > 3 * initial_step_size
            )
            full_turn = (
                has_left_start
                and distance_from_start < 3 * step_size
                and candidate[1] > axis[1]
            )
            out_of_bounds = not self._in_bounds(candidate)
            if full_turn or out_of_bounds:
                step_size *= 0.5
            else:
                point = candidate

            if step_size < 1e-5:
                return full_turn

        return False