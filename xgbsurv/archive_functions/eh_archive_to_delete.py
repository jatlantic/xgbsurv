
def eh_gradient_2(
    # y and linear predictor contain two cols
    y: np.array,
    linear_predictor: np.array,
    sample_weight: np.array = 1.0,
) -> np.array:
    # XGBoost limitation
    y1 = y[:, 0]
    time, event = transform_back(y1)
    n_samples: int = time.shape[0]
    bandwidth = 1.30 * math.pow(n_samples, -0.2)
    linear_predictor_1: np.array = np.exp(linear_predictor[:, 0] * sample_weight)
    linear_predictor_2: np.array = np.exp(linear_predictor[:, 1] * sample_weight)
    linear_predictor_misc = np.log(time * linear_predictor_1)
    linear_predictor_vanilla: np.array = linear_predictor_2 / linear_predictor_1
    # call this R for consistency with formula
    #linear_predictor = np.log(time * linear_predictor_1)
    n_events: int = np.sum(event)
    gradient: np.array = np.empty(n_samples)
    event_mask: np.array = event.astype(np.bool_)
    inverse_sample_size: float = 1 / n_samples
    inverse_bandwidth: float = 1 / bandwidth
    zero_kernel: float = PDF_PREFACTOR
    zero_integrated_kernel: float = CDF_ZERO
    event_count: int = 0

    (
        difference_outer_product,
        kernel_matrix,
        integrated_kernel_matrix,
    ) = difference_kernels(
        a=linear_predictor_misc, b=linear_predictor_misc[event_mask], bandwidth=bandwidth
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

    integrated_kernel_denominator: np.array = (
        integrated_kernel_matrix * sample_repeated_linear_predictor
    ).sum(axis=0)

    for _ in range(n_samples):
        sample_event: int = event[_]
        gradient_five = -(
            inverse_sample_size
            * (
                (
                    linear_predictor_vanilla[_]
                    * integrated_kernel_matrix[_, :]
                    #* inverse_bandwidth
                )
                / integrated_kernel_denominator
            ).sum()
        )

        print('gradient_five',gradient_five)
        gradient_correction_factor = -(inverse_sample_size * (
                (
                    linear_predictor_vanilla[_] * zero_integrated_kernel
                    #* inverse_bandwidth
                )
                / integrated_kernel_denominator[event_count]
            )
            )
        print('gradient_correction_factor',gradient_correction_factor)

        if sample_event:
            
            print('inverse_sample_size', inverse_sample_size)
            gradient[_] = (

                + gradient_five
                )  + inverse_sample_size
            
            event_count += 1

        else:
            gradient[_] = gradient_five

    return np.negative(gradient)


#@jit(nopython=True, cache=True, fastmath=True)
def eh_gradient(
    # y and linear predictor contain two cols
    y: np.array,
    linear_predictor: np.array,
    sample_weight: np.array = 1.0,
) -> np.array:
    # XGBoost limitation
    y1 = y[:, 0]
    time, event = transform_back(y1)
    n_samples: int = time.shape[0]
    bandwidth = 1.30 * math.pow(n_samples, -0.2)
    linear_predictor_1: np.array = np.exp(linear_predictor[:, 0] * sample_weight)
    linear_predictor_2: np.array = np.exp(linear_predictor[:, 1] * sample_weight)
    linear_predictor_misc = np.log(time * linear_predictor_1)
    linear_predictor_vanilla: np.array = linear_predictor_2 / linear_predictor_1
    # call this R for consistency with formula
    #linear_predictor = np.log(time * linear_predictor_1)
    n_events: int = np.sum(event)
    gradient: np.array = np.empty(n_samples)
    event_mask: np.array = event.astype(np.bool_)
    inverse_sample_size: float = 1 / n_samples
    inverse_bandwidth: float = 1 / bandwidth
    zero_kernel: float = PDF_PREFACTOR
    zero_integrated_kernel: float = CDF_ZERO
    event_count: int = 0

    (
        difference_outer_product,
        kernel_matrix,
        integrated_kernel_matrix,
    ) = difference_kernels(
        a=linear_predictor_misc, b=linear_predictor_misc[event_mask], bandwidth=bandwidth
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

        gradient_five = -(
            inverse_sample_size
            * (
                (
                    linear_predictor_vanilla[_]
                    * integrated_kernel_matrix[_, :]
                    * inverse_bandwidth
                )
                / integrated_kernel_denominator
            ).sum()
        )



        if sample_event:
            gradient_correction_factor = (inverse_sample_size * (
                (
                    linear_predictor_vanilla[_] * zero_integrated_kernel
                    + linear_predictor_vanilla[_]
                    * zero_kernel
                    * inverse_bandwidth
                )
                / integrated_kernel_denominator[event_count]
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



            prefactor: float = kernel_numerator_full[
                event_mask, event_count
            ].sum() / (kernel_denominator[event_count])

            gradient_two = inverse_sample_size * prefactor

            # potentially part missing
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
                + gradient_one
                + gradient_two
                + gradient_three
                + gradient_four
                + gradient_correction_factor
                #+ gradient_five
                )  - inverse_sample_size
                #- inverse_sample_size # potentially remove this n/Delta_q
            
            event_count += 1

        else:
            gradient[_] = gradient_three

    return np.negative(gradient)