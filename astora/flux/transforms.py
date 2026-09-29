from abc import ABC, abstractmethod
from numpy import exp, ndarray, asarray, full_like, log, arctan, pi
from scipy.sparse import diags_array, sparray
from midas.parameters import ParameterVector


class FluxTransform(ABC):
    transform_parameters: ParameterVector
    
    @abstractmethod
    def transform(
        self,
        psi: ndarray,
        parameters: ndarray,
    ) -> ndarray:
        pass

    @abstractmethod
    def transform_and_jacobians(
        self,
        psi: ndarray,
        parameters: ndarray,
    ) -> tuple[ndarray, dict[str, ndarray | sparray]]:
        pass


class LogisticFlux(FluxTransform):
    def __init__(self, name: str):
        assert isinstance(name, str) and len(name) > 0
        self.transform_parameters = ParameterVector(name, 2)

    def transform(
        self,
        psi: ndarray,
        parameters: ndarray,
    ) -> ndarray:
        psi0, dpsi = parameters
        z = (psi - psi0) / dpsi
        return 1.0 / (1.0 + exp(-z))

    def transform_and_jacobians(
        self,
        psi: ndarray,
        parameters: ndarray,
    ) -> tuple[ndarray, dict[str, ndarray | sparray]]:
        psi0, dpsi = parameters
        z = (psi - psi0) / dpsi

        u = 1.0 / (1.0 + exp(-z))
        du_dz = u * (1.0 - u)

        jacobians = {
            "psi": diags_array(du_dz / dpsi),
            "parameters": du_dz[:, None] * asarray(
                (full_like(z, -1.0 / dpsi), -z / dpsi)
            ).T,
        }

        return u, jacobians


class CauchyCDF(FluxTransform):
    """Map flux to (0, 1) using parameters ``(psi0, ln_dpsi)``.

    The scale is chosen so the midpoint slope is ``1 / (4 * dpsi)``, matching
    :class:`LogisticFlux` for the same positive width ``dpsi = exp(ln_dpsi)``.
    """

    def __init__(self, name: str):
        assert isinstance(name, str) and len(name) > 0
        self.transform_parameters = ParameterVector(name, 2)

    def transform(
        self,
        psi: ndarray,
        parameters: ndarray,
    ) -> ndarray:
        psi0, ln_dpsi = parameters
        inverse_width = exp(-ln_dpsi)
        z = 0.25 * pi * (psi - psi0) * inverse_width
        return 0.5 + arctan(z) / pi

    def transform_and_jacobians(
        self,
        psi: ndarray,
        parameters: ndarray,
    ) -> tuple[ndarray, dict[str, ndarray | sparray]]:
        psi0, ln_dpsi = parameters
        inverse_width = exp(-ln_dpsi)
        z = 0.25 * pi * (psi - psi0) * inverse_width

        u = 0.5 + arctan(z) / pi
        du_dz = 1.0 / (pi * (1.0 + z**2))
        du_dpsi = 0.25 * inverse_width / (1.0 + z**2)

        jacobians = {
            "psi": diags_array(du_dpsi),
            "parameters": asarray(
                (
                    -du_dpsi,
                    -du_dz * z,
                )
            ).T,
        }

        return u, jacobians


class GenLogFlux(FluxTransform):
    def __init__(self, name: str):
        assert isinstance(name, str) and len(name) > 0
        self.transform_parameters = ParameterVector(name, 3)

    def transform(
        self,
        psi: ndarray,
        parameters: ndarray,
    ) -> ndarray:
        psi0, dpsi, ln_k = parameters
        k = exp(ln_k)
        w_corrected = dpsi * 2 * k * (1 - 2**(-1 / k))
        z = (psi - psi0) / w_corrected
        dz = -log(2**(1 / k) - 1)
        return (1 + exp(-(z + dz)))**-k

    def transform_and_jacobians(
        self,
        psi: ndarray,
        parameters: ndarray,
    ) -> tuple[ndarray, dict[str, ndarray | sparray]]:
        psi0, dpsi, ln_k = parameters

        k = exp(ln_k)
        two_to_minus_inv_k = 2**(-1 / k)
        one_minus_power = 1 - two_to_minus_inv_k
        w_corrected = dpsi * 2 * k * one_minus_power
        z = (psi - psi0) / w_corrected
        dz = -log(2**(1 / k) - 1)
        shifted_z = z + dz

        logistic = 1 / (1 + exp(-shifted_z))
        u = logistic**k
        du_dshifted_z = k * u * (1 - logistic)

        log_two_over_k_one_minus_power = log(2) / (k * one_minus_power)
        dlog_width_dln_k = (
            1 - two_to_minus_inv_k * log_two_over_k_one_minus_power
        )
        dshifted_z_dln_k = (
            -z * dlog_width_dln_k + log_two_over_k_one_minus_power
        )
        du_dln_k = u * k * (
            log(logistic) + (1 - logistic) * dshifted_z_dln_k
        )

        jacobians = {
            "psi": diags_array(du_dshifted_z / w_corrected),
            "parameters": asarray(
                (
                    -du_dshifted_z / w_corrected,
                    -du_dshifted_z * z / dpsi,
                    du_dln_k,
                )
            ).T,
        }

        return u, jacobians

    