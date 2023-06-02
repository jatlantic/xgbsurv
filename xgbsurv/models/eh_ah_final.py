from typing import Tuple, Callable
from xgbsurv.models.utils import transform, transform_back
import numpy as np
from numba import jit
from scipy.stats import norm
from sklearn.utils.extmath import safe_sparse_dot
from typeguard import typechecked
import pandas as pd
from scipy.integrate import quadrature

from math import exp, sqrt, pi, erf
from numba import jit
import math

PDF_PREFACTOR: float = 0.3989424488876037
SQRT_TWO: float = 1.4142135623730951
SQRT_EPS: float = 1.4901161193847656e-08
EPS: float = 2.220446049250313e-16
CDF_ZERO: float = 0.5


@jit(nopython=True, cache=True, fastmath=True)
def gaussian_integrated_kernel(x):
    return 0.5 * (1 + erf(x / SQRT_TWO))


@jit(nopython=True, cache=True, fastmath=True)
def gaussian_kernel(x):
    return PDF_PREFACTOR * exp(-0.5 * (x**2))


@jit(nopython=True, cache=True, fastmath=True)
def kernel(a, b, bandwidth):
    kernel_matrix: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    for ix in range(a.shape[0]):
        for qx in range(b.shape[0]):
            kernel_matrix[ix, qx] = gaussian_kernel(
                (a[ix] - b[qx]) / bandwidth
            )
    return kernel_matrix


@jit(nopython=True, cache=True, fastmath=True)
def integrated_kernel(a, b, bandwidth):
    integrated_kernel_matrix: np.array = np.empty(
        shape=(a.shape[0], b.shape[0])
    )
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
    integrated_kernel_matrix: np.array = np.empty(
        shape=(a.shape[0], b.shape[0])
    )
    for ix in range(a.shape[0]):
        for qx in range(b.shape[0]):
            difference[ix, qx] = (a[ix] - b[qx]) / bandwidth
            kernel_matrix[ix, qx] = gaussian_kernel(difference[ix, qx])
            integrated_kernel_matrix[ix, qx] = gaussian_integrated_kernel(
                difference[ix, qx]
            )

    return difference, kernel_matrix, integrated_kernel_matrix





@jit(nopython=True, cache=True, fastmath=True)
def ah_likelihood(
    y: np.array,
    linear_predictor: np.array, 
    sample_weight: np.array = 1.0,
) -> np.array:
    """Partial likelihood estimator for Accelerated Hazards model.

    Args:
        linear_predictor (np.array): Linear predictor of risk: `X @ coef`. Shape = (n_samples,).
        time (np.array): Array containing event/censoring times of shape = (n_samples,).
        event (np.array): Array containing binary event indicators of shape = (n_samples,).
        bandwidth_function (str, optional): _description_. Defaults to "jones_1990".

    Returns:
        np.array: Scalar value of mean partial log likelihood estimate.
    """
    time, event = transform_back(y)
    n_samples: int = time.shape[0]
    n_events = np.sum(event)
    bandwidth = 1.30 * math.pow(n_samples, -0.2)
    linear_predictor: np.array = linear_predictor
    # attention wrt. -exp
    exp_linear_predictor: np.array = np.exp(-linear_predictor)
    # this has to be separated
    R_linear_predictor: np.array = np.log(time * np.exp(linear_predictor))
    inverse_sample_size_bandwidth: float = 1 / (n_samples * bandwidth)
    event_mask: np.array = event.astype(np.bool_)

    _: np.array
    kernel_matrix: np.array
    integrated_kernel_matrix: np.array

    (_, kernel_matrix, integrated_kernel_matrix,) = difference_kernels(
        a=R_linear_predictor,
        b=R_linear_predictor[event_mask],
        bandwidth=bandwidth,
    )
    kernel_matrix = kernel_matrix[event_mask, :]

    inverse_sample_size: float = 1 / n_samples

    kernel_sum: np.array = kernel_matrix.sum(axis=0)

    integrated_kernel_sum: np.array = (
        integrated_kernel_matrix
        * exp_linear_predictor.repeat(np.sum(event)).reshape(-1, np.sum(event))
    ).sum(axis=0)
    likelihood: np.array = inverse_sample_size * (
        -R_linear_predictor[event_mask].sum()
        + np.log(inverse_sample_size_bandwidth * kernel_sum).sum()
        - np.log(inverse_sample_size * integrated_kernel_sum).sum()
    )
    return -likelihood*n_events # *n_samples


@jit(nopython=True, cache=True, fastmath=True)
def ah_objective(
    y: np.array,
    linear_predictor: np.array
) -> Tuple[np.array, np.array]:
    """Gradient of the Accelerated Hazards model in numba-compatible form.

    Args:
        time (np.array): Array containing event/censoring times of shape = (n_samples,).
        event (np.array): Array containing binary event indicators of shape = (n_samples,).
        linear_predictor (np.array): Linear predictor of risk: `X @ coef`. Shape = (n_samples,).
        bandwidth_function (str, optional): _description_. Defaults to "jones_1990".
        hessian_modification_strategy (str, optional): _description_. Defaults to "flip".

    Returns:
        Tuple[np.array, np.array]: Tuple containing the negative gradients and the hessian
            of with the linear predictor.
    """
    time, event = transform_back(y)
    n_samples: int = time.shape[0]
    bandwidth = 1.30 * math.pow(n_samples, -0.2)
    linear_predictor_vanilla: np.array = np.exp(-linear_predictor)
    linear_predictor = np.log(time * np.exp(linear_predictor))
    n_events: int = np.sum(event)
    gradient: np.array = np.empty(n_samples)
    hessian: np.array = np.empty(n_samples)
    event_mask: np.array = event.astype(np.bool_)
    inverse_sample_size: float = 1 / n_samples
    inverse_bandwidth: float = 1 / bandwidth
    squared_inverse_bandwidth: float = inverse_bandwidth**2

    zero_kernel: float = PDF_PREFACTOR
    zero_integrated_kernel: float = CDF_ZERO
    event_count: int = 0

    (
        difference_outer_product,
        kernel_matrix,
        integrated_kernel_matrix,
    ) = difference_kernels(
        a=linear_predictor, b=linear_predictor[event_mask], bandwidth=bandwidth
    )



    sample_repeated_linear_predictor: np.array = (
        linear_predictor_vanilla.repeat(n_events).reshape(
            (n_samples, n_events)
        )
    )

    kernel_numerator_full: np.array = (
        kernel_matrix * difference_outer_product * inverse_bandwidth
    )




    kernel_denominator: np.array = kernel_matrix[event_mask, :].sum(axis=0)
    #squared_kernel_denominator: np.array = np.square(kernel_denominator)

    integrated_kernel_denominator: np.array = (
        integrated_kernel_matrix * sample_repeated_linear_predictor
    ).sum(axis=0)

    for _ in range(n_samples):
        sample_event: int = event[_]
        gradient_three = -(
            inverse_sample_size
            * (
                (
                    -linear_predictor_vanilla[_]
                    * integrated_kernel_matrix[_, :]
                    + linear_predictor_vanilla[_]
                    * kernel_matrix[_, :]
                    * inverse_bandwidth
                )
                / integrated_kernel_denominator
            ).sum()
        )

        if sample_event:
            gradient_correction_factor = inverse_sample_size * (
                (
                    linear_predictor_vanilla[_] * zero_integrated_kernel
                    + linear_predictor_vanilla[_]
                    * zero_kernel
                    * inverse_bandwidth
                )
                / integrated_kernel_denominator[event_count]
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

            prefactor: float = kernel_numerator_full[
                event_mask, event_count
            ].sum() / (kernel_denominator[event_count])

            gradient_two = inverse_sample_size * prefactor


            prefactor = (
                (
                    (
                        linear_predictor_vanilla
                        * kernel_matrix[:, event_count]
                    ).sum()
                    - linear_predictor_vanilla[_] * zero_kernel
                )
                * inverse_bandwidth
                - (linear_predictor_vanilla[_] * zero_integrated_kernel)
            ) / integrated_kernel_denominator[event_count]
            gradient_four = inverse_sample_size * prefactor


            gradient[_] = (
                gradient_one
                + gradient_two
                + gradient_three
                + gradient_four
                + gradient_correction_factor
            ) - inverse_sample_size

            event_count += 1

        else:
            gradient[_] = gradient_three
     
    return np.negative(gradient)*n_events, np.ones(gradient.shape[0])


@jit(nopython=True, cache=True, fastmath=True)
def baseline_hazard_estimator_ah(
    time,
    time_train,
    event_train,
    predictor_train,
):
    n_samples: int = time_train.shape[0]
    bandwidth = 1.30 * math.pow(n_samples, -0.2)
    inverse_bandwidth: float = 1 / bandwidth
    inverse_sample_size: float = 1 / n_samples
    inverse_bandwidth_sample_size: float = (
        inverse_sample_size * (1 / (time + EPS)) * inverse_bandwidth
    )
    # removed this log_time: float = time and added log
    log_time: float = np.log(time + EPS) # added this 24.05.23
    R_lp: np.array = np.log(time_train * np.exp(predictor_train))
    difference_lp_log_time: np.array = (R_lp - log_time) / bandwidth
    numerator: float = 0.0
    denominator: float = 0.0
    for _ in range(n_samples):
        difference: float = difference_lp_log_time[_]
        # added [_] for predictor
        denominator += np.exp(-predictor_train[_]) * gaussian_integrated_kernel(difference)
        if event_train[_]:
            numerator += gaussian_kernel(difference)
    numerator = inverse_bandwidth_sample_size * numerator
    denominator = inverse_sample_size * denominator
    if denominator <= 0.0:
        return 0.0
    else:
        return numerator / denominator

# latest version
# TODO: simplify inputs
def get_cumulative_hazard_function_ah(
    X_train, 
    X_test, 
    y_train, 
    y_test,
    predictor_train,
    predictor_test,
):
    time_test, event_test = transform_back(y_test)
    time: np.array = np.unique(time_test)
    time_train, event_train = transform_back(y_train)
    theta: np.array = np.exp(predictor_test)
    n_samples: int = predictor_test.shape[0]

    zero_flag: bool = False
    if 0 not in time:
        zero_flag = True
        time = np.concatenate([np.array([0]), time])
        cumulative_hazard: np.array = np.empty((n_samples, time.shape[0]))
    else:
        cumulative_hazard: np.array = np.empty((n_samples, time.shape[0]))

    def hazard_function_integrate(s):
        return baseline_hazard_estimator_ah(
            time=s,
            time_train=time_train,
            event_train=event_train,
            predictor_train=predictor_train,
        )

    integration_times = np.stack(
        [
            np.unique(
                np.ravel(y_test)[
                    np.ravel((np.abs(y_test) == y_test).astype(np.bool_))
                ]
            )
            * i
            for i in np.round(np.exp(np.ravel(predictor_test)), 2)
        ]
    )
    integration_times = np.unique((np.ravel(integration_times)))

    integration_times = np.concatenate([[0], integration_times, [np.max(integration_times) + 0.01]])

    integration_values = np.zeros(integration_times.shape[0])
    for _ in range(1, integration_values.shape[0]):
        integration_values[_] = (
            integration_values[_ - 1]
            + quadrature(
                func=hazard_function_integrate,
                a=integration_times[_ - 1],
                b=integration_times[_],
                vec_func=False,
            )[0]
        )
    for _ in range(n_samples):
        cumulative_hazard[_] = (
            integration_values[
                np.digitize(
                    x=time * theta[_], bins=integration_times, right=False
                )
                - 1
            ]
            / theta[_]
        )
    if zero_flag:
        cumulative_hazard = cumulative_hazard[:, 1:]
        time = time[1:]
    return pd.DataFrame(cumulative_hazard, columns=time).T.sort_index(axis=0)



# formulation with hessian


# @jit(nopython=True, cache=True, fastmath=True)
# def modify_hessian(hessian: np.array):
#     if np.any(hessian < 0):
#         hessian[hessian < 0] = np.mean(hessian[hessian > 0])
#     return hessian

# @jit(nopython=True, cache=True, fastmath=True)
# def ah_objective(
#     y: np.array,
#     linear_predictor: np.array
# ) -> Tuple[np.array, np.array]:
#     """Gradient of the Accelerated Hazards model in numba-compatible form.

#     Args:
#         time (np.array): Array containing event/censoring times of shape = (n_samples,).
#         event (np.array): Array containing binary event indicators of shape = (n_samples,).
#         linear_predictor (np.array): Linear predictor of risk: `X @ coef`. Shape = (n_samples,).
#         bandwidth_function (str, optional): _description_. Defaults to "jones_1990".
#         hessian_modification_strategy (str, optional): _description_. Defaults to "flip".

#     Returns:
#         Tuple[np.array, np.array]: Tuple containing the negative gradients and the hessian
#             of with the linear predictor.
#     """
#     time, event = transform_back(y)
#     n_samples: int = time.shape[0]
#     bandwidth = 1.30 * math.pow(n_samples, -0.2)
#     linear_predictor_vanilla: np.array = np.exp(-linear_predictor)
#     linear_predictor = np.log(time * np.exp(linear_predictor))
#     n_events: int = np.sum(event)
#     gradient: np.array = np.empty(n_samples)
#     hessian: np.array = np.empty(n_samples)
#     event_mask: np.array = event.astype(np.bool_)
#     inverse_sample_size: float = 1 / n_samples
#     inverse_bandwidth: float = 1 / bandwidth
#     squared_inverse_bandwidth: float = inverse_bandwidth**2

#     zero_kernel: float = PDF_PREFACTOR
#     zero_integrated_kernel: float = CDF_ZERO
#     event_count: int = 0

#     (
#         difference_outer_product,
#         kernel_matrix,
#         integrated_kernel_matrix,
#     ) = difference_kernels(
#         a=linear_predictor, b=linear_predictor[event_mask], bandwidth=bandwidth
#     )

#     squared_difference_outer_product: np.array = np.square(
#         difference_outer_product
#     )

#     sample_repeated_linear_predictor: np.array = (
#         linear_predictor_vanilla.repeat(n_events).reshape(
#             (n_samples, n_events)
#         )
#     )

#     kernel_numerator_full: np.array = (
#         kernel_matrix * difference_outer_product * inverse_bandwidth
#     )
#     squared_kernel_numerator: np.array = np.square(
#         kernel_numerator_full[event_mask, :]
#     )

#     squared_difference_kernel_numerator: np.array = kernel_matrix[
#         event_mask, :
#     ] * (
#         squared_difference_outer_product[event_mask, :]
#         * squared_inverse_bandwidth
#     )

#     kernel_denominator: np.array = kernel_matrix[event_mask, :].sum(axis=0)
#     squared_kernel_denominator: np.array = np.square(kernel_denominator)

#     integrated_kernel_denominator: np.array = (
#         integrated_kernel_matrix * sample_repeated_linear_predictor
#     ).sum(axis=0)

#     for _ in range(n_samples):
#         sample_event: int = event[_]
#         gradient_three = -(
#             inverse_sample_size
#             * (
#                 (
#                     -linear_predictor_vanilla[_]
#                     * integrated_kernel_matrix[_, :]
#                     + linear_predictor_vanilla[_]
#                     * kernel_matrix[_, :]
#                     * inverse_bandwidth
#                 )
#                 / integrated_kernel_denominator
#             ).sum()
#         )

#         hessian_five = inverse_sample_size * (
#             (
#                 np.square(
#                     (
#                         linear_predictor_vanilla[_]
#                         * integrated_kernel_matrix[_, :]
#                         + linear_predictor_vanilla[_]
#                         * kernel_matrix[_, :]
#                         * inverse_bandwidth
#                     )
#                     / integrated_kernel_denominator
#                 )
#             ).sum()
#         )
#         hessian_six = -(
#             inverse_sample_size
#             * (
#                 (
#                     linear_predictor_vanilla[_]
#                     * integrated_kernel_matrix[_, :]
#                     + 2
#                     * linear_predictor_vanilla[_]
#                     * kernel_matrix[_, :]
#                     * inverse_bandwidth
#                     - linear_predictor_vanilla[_]
#                     * kernel_numerator_full[_, :]
#                     * inverse_bandwidth
#                 )
#                 / integrated_kernel_denominator
#             ).sum()
#         )

#         if sample_event:
#             gradient_correction_factor = inverse_sample_size * (
#                 (
#                     linear_predictor_vanilla[_] * zero_integrated_kernel
#                     + linear_predictor_vanilla[_]
#                     * zero_kernel
#                     * inverse_bandwidth
#                 )
#                 / integrated_kernel_denominator[event_count]
#             )

#             hessian_correction_factor = -inverse_sample_size * (
#                 (
#                     (
#                         linear_predictor_vanilla[_] * zero_integrated_kernel
#                         + linear_predictor_vanilla[_]
#                         * zero_kernel
#                         * inverse_bandwidth
#                     )
#                     / integrated_kernel_denominator[event_count]
#                 )
#                 ** 2
#                 - (
#                     (
#                         linear_predictor_vanilla[_] * zero_integrated_kernel
#                         + 2
#                         * linear_predictor_vanilla[_]
#                         * zero_kernel
#                         * inverse_bandwidth
#                     )
#                     / (integrated_kernel_denominator[event_count])
#                 )
#             )

#             gradient_one = -(
#                 inverse_sample_size
#                 * (
#                     kernel_numerator_full[
#                         _,
#                     ]
#                     / kernel_denominator
#                 ).sum()
#             )
#             hessian_one = -(
#                 inverse_sample_size
#                 * (
#                     squared_kernel_numerator[
#                         event_count,
#                     ]
#                     / squared_kernel_denominator
#                 ).sum()
#             )

#             hessian_two = inverse_sample_size * (
#                 (
#                     (
#                         squared_difference_kernel_numerator[event_count, :]
#                         - (
#                             kernel_matrix[
#                                 _,
#                             ]
#                             * squared_inverse_bandwidth
#                         )
#                     )
#                     / kernel_denominator
#                 ).sum()
#                 + (
#                     zero_kernel
#                     * squared_inverse_bandwidth
#                     / kernel_denominator[event_count]
#                 )
#             )

#             prefactor: float = kernel_numerator_full[
#                 event_mask, event_count
#             ].sum() / (kernel_denominator[event_count])

#             gradient_two = inverse_sample_size * prefactor
#             hessian_three = -inverse_sample_size * (prefactor**2)

#             hessian_four = inverse_sample_size * (
#                 (
#                     (
#                         (
#                             squared_difference_kernel_numerator[:, event_count]
#                         ).sum()
#                     )
#                     - (
#                         squared_inverse_bandwidth
#                         * (
#                             (kernel_matrix[event_mask, event_count]).sum()
#                             - zero_kernel
#                         )
#                     )
#                 )
#                 / (kernel_denominator[event_count])
#             )
#             prefactor = (
#                 (
#                     (
#                         linear_predictor_vanilla
#                         * kernel_matrix[:, event_count]
#                     ).sum()
#                     - linear_predictor_vanilla[_] * zero_kernel
#                 )
#                 * inverse_bandwidth
#                 - (linear_predictor_vanilla[_] * zero_integrated_kernel)
#             ) / integrated_kernel_denominator[event_count]
#             gradient_four = inverse_sample_size * prefactor

#             hessian_seven = inverse_sample_size * (prefactor**2)
#             hessian_eight = inverse_sample_size * (
#                 (
#                     (
#                         linear_predictor_vanilla
#                         * kernel_numerator_full[:, event_count]
#                         * inverse_bandwidth
#                     ).sum()
#                     - linear_predictor_vanilla[_] * zero_integrated_kernel
#                 )
#                 / integrated_kernel_denominator[event_count]
#             )

#             gradient[_] = (
#                 gradient_one
#                 + gradient_two
#                 + gradient_three
#                 + gradient_four
#                 + gradient_correction_factor
#             ) - inverse_sample_size

#             hessian[_] = (
#                 hessian_one
#                 + hessian_two
#                 + hessian_three
#                 + hessian_four
#                 + hessian_five
#                 + hessian_six
#                 + hessian_seven
#                 + hessian_eight
#                 + hessian_correction_factor
#             )
#             event_count += 1

#         else:
#             gradient[_] = gradient_three
#             hessian[_] = hessian_five + hessian_six
#     # setting the hessian here to one does not seem to work.        
#     return np.negative(gradient)* n_samples, np.ones(gradient.shape[0]) #modify_hessian(hessian=np.negative(hessian))
