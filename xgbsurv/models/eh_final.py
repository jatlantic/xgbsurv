# Extended Hazards final
from math import log # change below
import numpy as np
from numba import jit
import numpy.typing as npt
from math import exp, sqrt, pi, erf
from xgbsurv.models.utils import transform, transform_back

PDF_PREFACTOR: float = 0.3989424488876037
SQRT_TWO: float = 1.4142135623730951
SQRT_EPS: float = 1.4901161193847656e-08
CDF_ZERO: float = 0.5

#@jit(nopython=True, cache=True, fastmath=True)
def bandwidth_function(time, event, n):
    return (8*(sqrt(2)/3))**(1/5)*n**(-1/5)

#@jit(nopython=True, cache=True, fastmath=True)
def gaussian_integrated_kernel(x):
    return 0.5 * (1 + erf(x / SQRT_TWO))


#@jit(nopython=True, cache=True, fastmath=True)
def gaussian_kernel(x):
    # return (1 / sqrt(2 * 3.14159)) * exp(-1 / 2 * (x**2))
    return PDF_PREFACTOR * exp(-0.5 * (x**2))


#@jit(nopython=True, cache=True, fastmath=True)
def kernel(a, b, bandwidth):
    kernel_matrix: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    # intermediate_result: np.array = np.subtract.outer(a, b) / bandwidth
    for ix in range(a.shape[0]):
        for qx in range(b.shape[0]):
            kernel_matrix[ix, qx] = gaussian_kernel((a[ix] - b[qx]) / bandwidth)
    return kernel_matrix


#@jit(nopython=True, cache=True, fastmath=True)
def integrated_kernel(a, b, bandwidth):
    integrated_kernel_matrix: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    # intermediate_result: np.array = np.subtract.outer(a, b) / bandwidth
    for ix in range(a.shape[0]):
        for qx in range(b.shape[0]):
            integrated_kernel_matrix[ix, qx] = gaussian_integrated_kernel(
                (a[ix] - b[qx]) / bandwidth
            )
    return integrated_kernel_matrix



#@jit(nopython=True, cache=True, fastmath=True)
def eh_likelihood(
    # y and linear predictor contain two cols
    y: np.array, 
    linear_predictor: np.array,
    sample_weight: np.array = 1.0,
) -> np.array:
    
    # need two predictors here
    linear_predictor_1: np.array = linear_predictor[:,0]
    linear_predictor_2: np.array = linear_predictor[:,1]
    time, event = transform_back(y)
    n_samples: int = time.shape[0]
    bandwidth: float = bandwidth_function(time=time, event=event, n=n_samples)
    linear_predictor_1: np.array = linear_predictor_1 * sample_weight
    R_linear_predictor_1: np.array = np.log(time * np.exp(linear_predictor_1))
    inverse_sample_size_bandwidth: float = 1 / (n_samples * bandwidth)
    event_mask: np.array = event.astype(np.bool_)
    _: np.array
    kernel_matrix: np.array
    integrated_kernel_matrix: np.array

    (_, kernel_matrix, integrated_kernel_matrix,) = difference_kernels_eh(
        a=linear_predictor_1, b=linear_predictor_1[event_mask], bandwidth=bandwidth, 
        h1=linear_predictor_1, h2=linear_predictor_2
    )
    kernel_matrix = kernel_matrix[event_mask, :]

    inverse_sample_size: float = 1 / n_samples
    kernel_sum: np.array = kernel_matrix.sum(axis=0)
    # integrated kernel sum adapted to eh!
    integrated_kernel_sum: np.array = integrated_kernel_matrix.sum(0)

    likelihood: np.array = inverse_sample_size * (
        linear_predictor_2[event_mask].sum()
        - R_linear_predictor_1[event_mask].sum()
        + np.log(inverse_sample_size_bandwidth * kernel_sum).sum()
        - np.log(inverse_sample_size * integrated_kernel_sum).sum()
    )
    return -likelihood



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


#@jit(nopython=True, cache=True, fastmath=True)
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

#@jit(nopython=True, cache=True, fastmath=True)
def difference_kernels_eh(a, b, bandwidth, h1, h2):
    difference: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    kernel_matrix_eh: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    integrated_kernel_matrix_eh: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    # intermediate_result: np.array = np.subtract.outer(a, b) / bandwidth
    for ix in range(a.shape[0]):
        for qx in range(b.shape[0]): # all events as inputs
            difference[ix, qx] = (a[ix] - b[qx]) / bandwidth
            kernel_matrix_eh[ix, qx] = (np.exp(h2[qx])/np.exp(h1[qx])) * gaussian_kernel(difference[ix, qx])
            # put eh factor in
            integrated_kernel_matrix_eh[ix, qx] = (np.exp(h2[qx])/np.exp(h1[qx])) * gaussian_integrated_kernel(
                difference[ix, qx]
            )

    return difference, kernel_matrix_eh, integrated_kernel_matrix_eh





    

@jit(nopython=True, cache=True, fastmath=True)
def eh_objective(
    y: np.array, 
    linear_predictor: np.array,
    sample_weight: np.array = 1.0,
    #bandwidth: float,
    hessian_modification_strategy: str = "eps",
):
    # need two predictors here
    linear_predictor_1: np.array = linear_predictor[:,0]
    linear_predictor_2: np.array = linear_predictor[:,1]
    time, event = transform_back(y)
    linear_predictor: np.array = np.exp(sample_weight * linear_predictor)
    linear_predictor = np.log(time * linear_predictor)
    n_samples: int = time.shape[0]
    bandwidth: float = bandwidth_function(time=time, event=event, n=n_samples)
    #print('bandwidth', bandwidth)
    gradient: np.array = np.empty(n_samples)
    hessian: np.array = np.empty(n_samples)
    event_mask: np.array = event.astype(np.bool_)
    inverse_sample_size: float = 1 / n_samples
    inverse_bandwidth: float = 1 / bandwidth
    squared_inverse_bandwidth: float = inverse_bandwidth**2
    inverse_sample_size_bandwidth: float = inverse_sample_size * inverse_bandwidth

    zero_kernel: float = PDF_PREFACTOR
    event_count: int = 0
    squared_zero_kernel: float = zero_kernel**2

    (
        difference_eh,
        kernel_matrix_eh,
        integrated_kernel_matrix_eh,
    ) = difference_kernels_eh(
        a=linear_predictor, b=linear_predictor[event_mask], h1 = linear_predictor_1, h2 = linear_predictor_2, bandwidth=bandwidth
    )

    (
        difference_outer_product, # naming?
        kernel_matrix,
        integrated_kernel_matrix,
    ) = difference_kernels(
        a=linear_predictor, b=linear_predictor[event_mask], bandwidth=bandwidth
    )

    squared_kernel_matrix: np.array = np.square(kernel_matrix)
    squared_difference_outer_product: np.array = np.square(difference_outer_product)

    kernel_numerator_full: np.array = (
        kernel_matrix * difference_outer_product * inverse_bandwidth
    )
    squared_kernel_numerator: np.array = np.square(kernel_numerator_full[event_mask, :])

    squared_difference_kernel_numerator: np.array = kernel_matrix[event_mask, :] * (
        squared_difference_outer_product[event_mask, :] * squared_inverse_bandwidth
    )

    kernel_denominator: np.array = kernel_matrix[event_mask, :].sum(axis=0)
    squared_kernel_denominator: np.array = np.square(kernel_denominator)

    integrated_kernel_denominator: np.array = integrated_kernel_matrix.sum(axis=0)
    integrated_kernel_denominator_eh: np.array = integrated_kernel_matrix_eh.sum(axis=0)
    squared_integrated_kernel_denominator: np.array = np.square(
        integrated_kernel_denominator
    )

    for _ in range(n_samples):

        sample_event: int = event[_]
        # needs to be adapted
        gradient_three = -(
            inverse_sample_size
            * (
                (kernel_matrix_eh[_, :] * inverse_bandwidth)-(integrated_kernel_matrix_eh) / integrated_kernel_denominator_eh
            ).sum()
        )
        gradient_five = -(
            inverse_sample_size
            * (
                integrated_kernel_matrix_eh / integrated_kernel_denominator_eh
            ).sum()
        )
        hessian_five = (
            inverse_sample_size
            * (
                squared_kernel_matrix[_, :]
                * squared_inverse_bandwidth
                / squared_integrated_kernel_denominator
            ).sum()
        )
        hessian_six = (
            inverse_sample_size
            * (
                kernel_numerator_full[_, :]
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

            hessian_correction_factor = -inverse_sample_size * (
                squared_zero_kernel
                * squared_inverse_bandwidth
                / squared_integrated_kernel_denominator[event_count]
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
            # what is the zero kernel here?
            prefactor = (
                (kernel_matrix_eh[:, event_count].sum() - zero_kernel)
                * inverse_bandwidth
                / integrated_kernel_matrix_eh[:, event_count].sum()
            )
            gradient_four = inverse_sample_size * prefactor

            hessian_seven = inverse_sample_size * (prefactor**2)
            hessian_eight = inverse_sample_size * (
                (kernel_numerator_full[:, event_count] * inverse_bandwidth).sum()
                / integrated_kernel_denominator[event_count]
            )

            gradient[_] = (
                gradient_one
                + gradient_two
                + gradient_three
                + gradient_four
                + gradient_five
                + gradient_correction_factor
            )
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

    # print('gradient', gradient)
    # print('hessian', modify_hessian(
    #     hessian=np.negative(hessian),
    #     hessian_modification_strategy=hessian_modification_strategy
    # ))
    # hessian = modify_hessian(
    #     hessian=np.negative(hessian),
    #     hessian_modification_strategy=hessian_modification_strategy
    # )
    return  np.negative(gradient), np.ones(gradient.shape)