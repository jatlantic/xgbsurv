from typing import Tuple, Callable
from xgbsurv.models.utils import transform, transform_back
import numpy as np
from numba import jit
from scipy.stats import norm
from sklearn.utils.extmath import safe_sparse_dot
from typeguard import typechecked

from math import exp, sqrt, pi, erf
from numba import jit

PDF_PREFACTOR: float = 0.3989424488876037
SQRT_TWO: float = 1.4142135623730951
SQRT_EPS: float = 1.4901161193847656e-08
CDF_ZERO: float = 0.5

@jit(nopython=True, cache=True, fastmath=True)
def bandwidth_function(time, event, n):
    b = (8*(sqrt(2)/3))**(1/5)*n**(-1/5)
    #print(b)
    return b

@jit(nopython=True, cache=True, fastmath=True)
def gaussian_integrated_kernel(x):
    return 0.5 * (1 + erf(x / SQRT_TWO))


@jit(nopython=True, cache=True, fastmath=True)
def gaussian_kernel(x):
    # return (1 / sqrt(2 * 3.14159)) * exp(-1 / 2 * (x**2))
    return PDF_PREFACTOR * exp(-0.5 * (x**2))


@jit(nopython=True, cache=True, fastmath=True)
def kernel(a, b, bandwidth):
    kernel_matrix: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    # intermediate_result: np.array = np.subtract.outer(a, b) / bandwidth
    for ix in range(a.shape[0]):
        for qx in range(b.shape[0]):
            kernel_matrix[ix, qx] = gaussian_kernel((a[ix] - b[qx]) / bandwidth)
    return kernel_matrix


@jit(nopython=True, cache=True, fastmath=True)
def integrated_kernel(a, b, bandwidth):
    integrated_kernel_matrix: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    # intermediate_result: np.array = np.subtract.outer(a, b) / bandwidth
    for ix in range(a.shape[0]):
        for qx in range(b.shape[0]):
            integrated_kernel_matrix[ix, qx] = gaussian_integrated_kernel(
                (a[ix] - b[qx]) / bandwidth
            )
    return integrated_kernel_matrix


@jit(nopython=True, cache=True, fastmath=True)
def difference_kernels(a, b, bandwidth):
    difference: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    kernel_matrix: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    integrated_kernel_matrix: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    # intermediate_result: np.array = np.subtract.outer(a, b) / bandwidth
    for ix in range(a.shape[0]):
        for qx in range(b.shape[0]):
            difference[ix, qx] = (a[ix] - b[qx]) / bandwidth
            kernel_matrix[ix, qx] = gaussian_kernel(difference[ix, qx])
            integrated_kernel_matrix[ix, qx] = gaussian_integrated_kernel(
                difference[ix, qx]
            )

    return difference, kernel_matrix, integrated_kernel_matrix


@jit(nopython=True, cache=True, fastmath=True)
def modify_hessian(hessian: np.array, hessian_modification_strategy: str):
    if not np.any(hessian < 0):
        return hessian
    if hessian_modification_strategy == "ignore":
        hessian[hessian < 0] = 0
    elif hessian_modification_strategy == "eps":
        hessian[hessian < 0] = SQRT_EPS
    elif hessian_modification_strategy == "flip":
        hessian[hessian < 0] = np.abs(hessian[hessian < 0])
    elif hessian_modification_strategy is None:
        return hessian
    else:
        raise ValueError(
            "Expected `hessian_modification_strategy` to be one of ['ignore', 'eps', 'flip']."
            + f"Found {hessian_modification_strategy} instead."
        )
    return hessian


@jit(nopython=True, cache=True, fastmath=True)
def ah_likelihood(
    y: np.array,
    linear_predictor: np.array,
    sample_weight: np.array = 1.0,
):
    time, event = transform_back(y)
    n_samples: int = time.shape[0]
    bandwidth: float = bandwidth_function(time=time, event=event, n=n_samples)
    linear_predictor: np.array = linear_predictor * sample_weight
    exp_linear_predictor: np.array = np.exp(linear_predictor)
    R_linear_predictor: np.array = np.log(time * exp_linear_predictor)
    inverse_sample_size_bandwidth: float = 1 / (n_samples * bandwidth)
    event_mask: np.array = event.astype(np.bool_)
    bandwidth: float = bandwidth_function(time=time, event=event, n=n_samples)

    _: np.array
    kernel_matrix: np.array
    integrated_kernel_matrix: np.array

    (_, kernel_matrix, integrated_kernel_matrix,) = difference_kernels(
        a=linear_predictor, b=linear_predictor[event_mask], bandwidth=bandwidth
    )
    kernel_matrix = kernel_matrix[event_mask, :]

    inverse_sample_size: float = 1 / n_samples

    kernel_sum: np.array = kernel_matrix.sum(axis=0)

    integrated_kernel_sum: np.array = (
        integrated_kernel_matrix
        * exp_linear_predictor.repeat(np.sum(event)).reshape(-1, np.sum(event))
    ).sum(axis=0)

    likelihood: np.array = inverse_sample_size * (
        -R_linear_predictor[event].sum()
        + np.log(inverse_sample_size_bandwidth * kernel_sum).sum()
        - np.log(inverse_sample_size * integrated_kernel_sum).sum()
    )
    return -likelihood

@jit(nopython=True, cache=True, fastmath=True)
def ah_objective(
    y: np.array,
    linear_predictor: np.array,
    sample_weight: np.array = 1.0,
    hessian_modification_strategy: str = "ignore",
):
    time, event = transform_back(y)
    n_samples: int = time.shape[0]
    bandwidth: float = bandwidth_function(time=time, event=event, n=n_samples)
    linear_predictor_vanilla: np.array = np.exp(sample_weight * linear_predictor)
    linear_predictor = np.log(time * linear_predictor_vanilla)
    squared_linear_predictor_vanilla: np.array = np.square(linear_predictor_vanilla)
    n_samples: int = time.shape[0]
    n_events: int = np.sum(event)
    gradient: np.array = np.empty(n_samples)
    hessian: np.array = np.empty(n_samples)
    event_mask: np.array = event.astype(np.bool_)
    inverse_sample_size: float = 1 / n_samples
    inverse_bandwidth: float = 1 / bandwidth
    squared_inverse_bandwidth: float = inverse_bandwidth**2
    inverse_sample_size_bandwidth: float = inverse_sample_size * inverse_bandwidth

    zero_kernel: float = 0.39894245
    zero_integrated_kernel: float = CDF_ZERO
    event_count: int = 0
    squared_zero_kernel: float = zero_kernel**2

    (
        difference_outer_product,
        kernel_matrix,
        integrated_kernel_matrix,
    ) = difference_kernels(
        a=linear_predictor, b=linear_predictor[event_mask], bandwidth=bandwidth
    )

    squared_kernel_matrix: np.array = np.square(kernel_matrix)
    squared_difference_outer_product: np.array = np.square(difference_outer_product)

    squared_integrated_kernel_matrix: np.array = np.square(integrated_kernel_matrix)

    sample_repeated_linear_predictor: np.array = linear_predictor_vanilla.repeat(
        n_events
    ).reshape((n_samples, n_events))
    # print(sample_repeated_linear_predictor)
    squared_sample_repeated_linear_predictor: np.array = (
        squared_linear_predictor_vanilla.repeat(n_events).reshape((n_samples, n_events))
    )

    kernel_numerator_full: np.array = (
        kernel_matrix * difference_outer_product * inverse_bandwidth
    )
    squared_kernel_numerator: np.array = np.square(kernel_numerator_full[event_mask, :])

    squared_difference_kernel_numerator: np.array = kernel_matrix[event_mask, :] * (
        squared_difference_outer_product[event_mask, :] * squared_inverse_bandwidth
    )

    kernel_denominator: np.array = kernel_matrix[event_mask, :].sum(axis=0)
    squared_kernel_denominator: np.array = np.square(kernel_denominator)

    integrated_kernel_denominator: np.array = (
        integrated_kernel_matrix * sample_repeated_linear_predictor
    ).sum(axis=0)
    squared_integrated_kernel_denominator: np.array = np.square(
        integrated_kernel_denominator
    )

    for _ in range(n_samples):

        sample_event: int = event[_]
        gradient_three = -(
            inverse_sample_size
            * (
                (
                    linear_predictor_vanilla[_] * integrated_kernel_matrix[_, :]
                    + linear_predictor_vanilla[_]
                    * kernel_matrix[_, :]
                    * inverse_bandwidth
                )
                / integrated_kernel_denominator
            ).sum()
        )

        hessian_five = inverse_sample_size * (
            (
                np.square(
                    (
                        linear_predictor_vanilla[_] * integrated_kernel_matrix[_, :]
                        + linear_predictor_vanilla[_]
                        * kernel_matrix[_, :]
                        * inverse_bandwidth
                    )
                    / integrated_kernel_denominator
                )
            ).sum()
        )
        hessian_six = -(
            inverse_sample_size
            * (
                (
                    linear_predictor_vanilla[_] * integrated_kernel_matrix[_, :]
                    + 2
                    * linear_predictor_vanilla[_]
                    * kernel_matrix[_, :]
                    * inverse_bandwidth
                    - linear_predictor_vanilla[_]
                    * kernel_numerator_full[_, :]
                    * inverse_bandwidth
                )
                / integrated_kernel_denominator
            ).sum()
        )

        if sample_event:
            gradient_correction_factor = inverse_sample_size * (
                (
                    linear_predictor_vanilla[_] * zero_integrated_kernel
                    + linear_predictor_vanilla[_] * zero_kernel * inverse_bandwidth
                )
                / integrated_kernel_denominator[event_count]
            )

            hessian_correction_factor = -inverse_sample_size * (
                (
                    (
                        linear_predictor_vanilla[_] * zero_integrated_kernel
                        + linear_predictor_vanilla[_] * zero_kernel * inverse_bandwidth
                    )
                    / integrated_kernel_denominator[event_count]
                )
                ** 2
                - (
                    (
                        linear_predictor_vanilla[_] * zero_integrated_kernel
                        + 2
                        * linear_predictor_vanilla[_]
                        * zero_kernel
                        * inverse_bandwidth
                    )
                    / (integrated_kernel_denominator[event_count])
                )
            )

            gradient_one = -(
                inverse_sample_size
                * (
                    kernel_numerator_full[
                        _,
                    ]
                    / kernel_denominator
                ).sum()
            )
            hessian_one = -(
                inverse_sample_size
                * (
                    squared_kernel_numerator[
                        event_count,
                    ]
                    / squared_kernel_denominator
                ).sum()
            )

            hessian_two = inverse_sample_size * (
                (
                    (
                        squared_difference_kernel_numerator[event_count, :]
                        - (
                            kernel_matrix[
                                _,
                            ]
                            * squared_inverse_bandwidth
                        )
                    )
                    / kernel_denominator
                ).sum()
                + (
                    zero_kernel
                    * squared_inverse_bandwidth
                    / kernel_denominator[event_count]
                )
            )

            prefactor: float = kernel_numerator_full[event_mask, event_count].sum() / (
                kernel_denominator[event_count]
            )

            gradient_two = inverse_sample_size * prefactor
            hessian_three = -inverse_sample_size * (prefactor**2)

            hessian_four = inverse_sample_size * (
                (
                    ((squared_difference_kernel_numerator[:, event_count]).sum())
                    - (
                        squared_inverse_bandwidth
                        * ((kernel_matrix[event_mask, event_count]).sum() - zero_kernel)
                    )
                )
                / (kernel_denominator[event_count])
            )
            prefactor = (
                (
                    (linear_predictor_vanilla * kernel_matrix[:, event_count]).sum()
                    - linear_predictor_vanilla[_] * zero_kernel
                )
                * inverse_bandwidth
                - (linear_predictor_vanilla[_] * zero_integrated_kernel)
            ) / integrated_kernel_denominator[event_count]
            gradient_four = inverse_sample_size * prefactor

            hessian_seven = inverse_sample_size * (prefactor**2)
            hessian_eight = inverse_sample_size * (
                (
                    (
                        linear_predictor_vanilla
                        * kernel_numerator_full[:, event_count]
                        * inverse_bandwidth
                    ).sum()
                    - linear_predictor_vanilla[_] * zero_integrated_kernel
                )
                / integrated_kernel_denominator[event_count]
            )

            gradient[_] = (
                gradient_one
                + gradient_two
                + gradient_three
                + gradient_four
                + gradient_correction_factor
            ) - inverse_sample_size
            hessian[_] = (
                hessian_one
                + hessian_two
                + hessian_three
                + hessian_four
                + hessian_five
                + hessian_six
                + hessian_seven
                + hessian_eight
                + hessian_correction_factor
            )
            event_count += 1

        else:
            gradient[_] = gradient_three
            hessian[_] = hessian_five + hessian_six
    # hessian = modify_hessian(
    #     hessian=np.negative(hessian),
    #     hessian_modification_strategy=hessian_modification_strategy,
    # ) 
    return np.negative(gradient), np.ones(gradient.shape)



class AhPredictor():
    """Prediction functions particular to the Cox PH model"""
    
    def __init__(self) -> None:
        self.uniq_times = None
        self.cum_hazard_baseline = None
        self.baseline_survival = None
        
    
    def fit(self, partial_hazard, y):
        raise NotImplementedError("This model does not provide for the function you asked for!")

    def get_cumulative_hazard_function(self):
        raise NotImplementedError("This model does not provide for the function you asked for!")

    def get_survival_function(self):
        raise NotImplementedError("This model does not provide for the function you asked for!")
