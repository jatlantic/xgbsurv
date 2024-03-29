from typing import Tuple, Callable
from xgbsurv.models.utils import transform, transform_back
import numpy as np
from numba import jit
import pandas as pd
from scipy.integrate import quadrature, cumtrapz
from functools import partial
from numba import cfunc
from scipy.stats import norm
from sklearn.utils.extmath import safe_sparse_dot
from typeguard import typechecked
from typing import Optional

from math import exp, sqrt, pi, erf, pow, log


PDF_PREFACTOR: float = 0.3989424488876037
SQRT_TWO: float = 1.4142135623730951
SQRT_EPS: float = 1.4901161193847656e-08
EPS: float = 2.220446049250313e-16
CDF_ZERO: float = 0.5

@jit(nopython=True, cache=True, fastmath=True)
def aft_likelihood(
    y: np.array,
    linear_predictor: np.array,
    sample_weight: float = 1.0,
    bandwidth: Optional[float] = None
) -> np.array:
    
    """Accelerated Failure Time Model's pseudo likelihood function.

    Parameters
    ----------
    y : np.array
        True values.
    linear_predictor : np.array
        Model predictions.
    sample_weight : float, optional
        Weight to set at your disgression, by default 1.0.
    bandwidth : float, optional
        Bandwidth constant, set by default.

    Returns
    -------
    np.array
        Pseudo likelihood.

    References
    ----------
    .. [1] Zhong, Q., Mueller, J. W. & Wang, J.-L. 
    Deep extended hazard models for survival analysis. 
    Advances in Neural Information Processing Systems 34, 15111–15124 (2021).

    .. [2] Tseng, Yi-Kuan, and Ken-Ning Shu. 2011. 
    “Efficient Estimation for a Semiparametric Extended Hazards Model.” 
    Communications in Statistics—Simulation and Computation® 40 (2): 258–73.

    """

    time, event = transform_back(y)
    if np.sum(event) == 0:
        raise RuntimeError("No events detected!")
    n_events = np.sum(event)
    n_samples: int = time.shape[0]
    if bandwidth==None:
        bandwidth = 1.30 * pow(n_samples, -0.2)
    linear_predictor: np.array = linear_predictor * sample_weight
    R_linear_predictor: np.array = np.log(time * np.exp(linear_predictor))
    inverse_sample_size_bandwidth: float = 1 / (n_samples * bandwidth)
    event_mask: np.array = event.astype(np.bool_)

    _: np.array
    kernel_matrix: np.array
    integrated_kernel_matrix: np.array

    (_, kernel_matrix, integrated_kernel_matrix,) = difference_kernels(
        a=R_linear_predictor, b=R_linear_predictor[event_mask], bandwidth=bandwidth
    )

    kernel_matrix = kernel_matrix[event_mask, :]

    inverse_sample_size: float = 1 / n_samples
    kernel_sum: np.array = kernel_matrix.sum(axis=0)
    integrated_kernel_sum: np.array = integrated_kernel_matrix.sum(0)

    likelihood: np.array = inverse_sample_size * (
        linear_predictor[event_mask].sum()
        - R_linear_predictor[event_mask].sum()
        + np.log(inverse_sample_size_bandwidth * kernel_sum).sum()
        - np.log(inverse_sample_size * integrated_kernel_sum).sum()
    )
    return -likelihood*n_events #*n_samples


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
def modify_hessian(hessian: np.array):
    if np.any(hessian < 0):
        hessian[hessian < 0] = np.mean(hessian[hessian > 0])
    return hessian


@jit(nopython=True, cache=True, fastmath=True)
def aft_objective(
    y: np.array, 
    linear_predictor: np.array, 
    sample_weight: float = 1.0
)->tuple[np.array, np.array]:
    """Objective function for Boosted Extendend Hazards Model.

    Parameters
    ----------
    y : np.array
        _description_
    linear_predictor : np.array
        _description_
    sample_weight : float, optional
        Sample weight, by default 1.0.
    Returns
    -------
    tuple[np.array, np.array]
        Tuple containing the negative gradients and the diagonal hessian
        of ones.
    """

    time, event = transform_back(y)
    n_events = np.sum(event)
    linear_predictor: np.array = np.exp(sample_weight * linear_predictor)
    linear_predictor = np.log(time * linear_predictor)
    n_samples: int = time.shape[0]
    bandwidth = 1.30 * pow(n_samples, -0.2)
    gradient: np.array = np.empty(n_samples)
    event_mask: np.array = event.astype(np.bool_)
    inverse_sample_size: float = 1 / n_samples
    inverse_bandwidth: float = 1 / bandwidth
    inverse_sample_size_bandwidth: float = (
        inverse_sample_size * inverse_bandwidth
    )

    zero_kernel: float = PDF_PREFACTOR
    event_count: int = 0

    (
        difference_outer_product,
        kernel_matrix,
        integrated_kernel_matrix,
    ) = difference_kernels(
        a=linear_predictor, b=linear_predictor[event_mask], bandwidth=bandwidth
    )

    kernel_numerator_full: np.array = (
        kernel_matrix * difference_outer_product * inverse_bandwidth
    )

    kernel_denominator: np.array = kernel_matrix[event_mask, :].sum(axis=0)

    integrated_kernel_denominator: np.array = integrated_kernel_matrix.sum(
        axis=0
    )


    for _ in range(n_samples):

        sample_event: int = event[_]
        gradient_three = -(
            inverse_sample_size
            * (
                kernel_matrix[_, :]
                * inverse_bandwidth
                / integrated_kernel_denominator
            ).sum()
        )


        if sample_event:
            gradient_correction_factor = (
                inverse_sample_size_bandwidth
                * zero_kernel
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
                (kernel_matrix[:, event_count].sum() - zero_kernel)
                * inverse_bandwidth
                / integrated_kernel_matrix[:, event_count].sum()
            )
            gradient_four = inverse_sample_size * prefactor


            gradient[_] = (
                gradient_one
                + gradient_two
                + gradient_three
                + gradient_four
                + gradient_correction_factor
            )

            event_count += 1

        else:
            gradient[_] = gradient_three

    grad = np.negative(gradient)
    return grad*n_events, np.ones(grad.shape[0]) #n_samples


@jit(nopython=True, cache=True, fastmath=True)
def aft_baseline_hazard_estimator(
    time, # unique test time, time to integrate over
    time_train,
    event_train,
    predictor_train,
):
    """Get Accelerated Failure Time Model's baseline hazard.

    Parameters
    ----------
    time : np.array
        _description_
    time_train : np.array
        _description_
    event_train : np.array
        _description_
    predictor_train : np.array
        _description_

    Returns
    -------
    np.array
        Baseline hazard.
    """
    n_samples: int = time_train.shape[0]
    bandwidth = 1.30 * pow(n_samples, -0.2)
    inverse_bandwidth: float = 1 / bandwidth
    inverse_sample_size: float = 1 / n_samples
    inverse_bandwidth_sample_size_time: float = (
        inverse_sample_size * (1 / (time + EPS)) * inverse_bandwidth
    )
    log_time: float = log(time + EPS)

    R_lp: np.array = np.log(time_train * np.exp(predictor_train))
    difference_lp_log_time: np.array = (R_lp - log_time) / bandwidth
    numerator: float = 0.0
    denominator: float = 0.0
    for _ in range(n_samples):
        difference: float = difference_lp_log_time[_]
        denominator += gaussian_integrated_kernel(difference)
        if event_train[_]:
            numerator += gaussian_kernel(difference)
    numerator = inverse_bandwidth_sample_size_time * numerator
    denominator = inverse_sample_size * denominator
    if denominator <= 0.0:
        return 0.0
    else:
        return numerator / denominator

#@jit(nopython=True, cache=True, fastmath=True)
def get_cumulative_hazard_function_aft(
    X_train,
    X_test,
    y_train,
    y_test,
    predictor_train,
    predictor_test,
)-> pd.DataFrame:
    
    """Get cumulative hazard function of Accelerated Failure Time Model.

    Parameters
    ----------
    X_train : np.array
        _description_
    X_test : np.array
        _description_
    y_train : np.array
        _description_
    y_test : np.array
        _description_
    predictor_train : np.array
        _description_
    predictor_test : np.array
        _description_

    Returns
    -------
    pd.DataFrame
        Cumulative hazard dataframe.
    """
    time_test, _ = transform_back(y_test)
    # changed to unique
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
        return aft_baseline_hazard_estimator(
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
        cumulative_hazard[_] = integration_values[
            np.digitize(x=time * theta[_], bins=integration_times, right=False)
            - 1
        ]
    if zero_flag:
        cumulative_hazard = cumulative_hazard[:, 1:]
        time = time[1:]
    return pd.DataFrame(cumulative_hazard, columns=time).T.sort_index(axis=0)




# version with hessian

# @jit(nopython=True, cache=True, fastmath=True)
# def aft_objective(
#     y: np.array, linear_predictor: np.array, sample_weight: np.array = 1.0
# ):
#     time, event = transform_back(y)
#     linear_predictor: np.array = np.exp(sample_weight * linear_predictor)
#     linear_predictor = np.log(time * linear_predictor)
#     #R_linear_predictor: np.array = np.log(time * np.exp(linear_predictor))
#     n_samples: int = time.shape[0]
#     bandwidth = 1.30 * pow(n_samples, -0.2)
#     gradient: np.array = np.empty(n_samples)
#     hessian: np.array = np.empty(n_samples)
#     event_mask: np.array = event.astype(np.bool_)
#     inverse_sample_size: float = 1 / n_samples
#     inverse_bandwidth: float = 1 / bandwidth
#     squared_inverse_bandwidth: float = inverse_bandwidth**2
#     inverse_sample_size_bandwidth: float = (
#         inverse_sample_size * inverse_bandwidth
#     )

#     zero_kernel: float = PDF_PREFACTOR
#     event_count: int = 0
#     squared_zero_kernel: float = zero_kernel**2

#     (
#         difference_outer_product,
#         kernel_matrix,
#         integrated_kernel_matrix,
#     ) = difference_kernels(
#         a=linear_predictor, b=linear_predictor[event_mask], bandwidth=bandwidth
#     )
    
#     # (
#     #     difference_outer_product,
#     #     kernel_matrix,
#     #     integrated_kernel_matrix,
#     # ) = difference_kernels(
#     #     a=R_linear_predictor, b=R_linear_predictor[event_mask], bandwidth=bandwidth
#     # )
#     squared_kernel_matrix: np.array = np.square(kernel_matrix)
#     squared_difference_outer_product: np.array = np.square(
#         difference_outer_product
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

#     integrated_kernel_denominator: np.array = integrated_kernel_matrix.sum(
#         axis=0
#     )
#     squared_integrated_kernel_denominator: np.array = np.square(
#         integrated_kernel_denominator
#     )

#     for _ in range(n_samples):

#         sample_event: int = event[_]
#         gradient_three = -(
#             inverse_sample_size
#             * (
#                 kernel_matrix[_, :]
#                 * inverse_bandwidth
#                 / integrated_kernel_denominator
#             ).sum()
#         )
#         hessian_five = (
#             inverse_sample_size
#             * (
#                 squared_kernel_matrix[_, :]
#                 * squared_inverse_bandwidth
#                 / squared_integrated_kernel_denominator
#             ).sum()
#         )
#         hessian_six = (
#             inverse_sample_size
#             * (
#                 kernel_numerator_full[_, :]
#                 * inverse_bandwidth
#                 / integrated_kernel_denominator
#             ).sum()
#         )

#         if sample_event:
#             gradient_correction_factor = (
#                 inverse_sample_size_bandwidth
#                 * zero_kernel
#                 / integrated_kernel_denominator[event_count]
#             )

#             hessian_correction_factor = -inverse_sample_size * (
#                 squared_zero_kernel
#                 * squared_inverse_bandwidth
#                 / squared_integrated_kernel_denominator[event_count]
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
#                 (kernel_matrix[:, event_count].sum() - zero_kernel)
#                 * inverse_bandwidth
#                 / integrated_kernel_matrix[:, event_count].sum()
#             )
#             gradient_four = inverse_sample_size * prefactor

#             hessian_seven = inverse_sample_size * (prefactor**2)
#             hessian_eight = inverse_sample_size * (
#                 (
#                     kernel_numerator_full[:, event_count] * inverse_bandwidth
#                 ).sum()
#                 / integrated_kernel_denominator[event_count]
#             )

#             gradient[_] = (
#                 gradient_one
#                 + gradient_two
#                 + gradient_three
#                 + gradient_four
#                 + gradient_correction_factor
#             )
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
#     grad = np.negative(gradient)
#     hess = modify_hessian(np.negative(hessian))
#     return grad*n_samples, np.ones(grad.shape[0]) #hess