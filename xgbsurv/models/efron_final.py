# Efron final
from math import log # change below
import numpy as np
from numba import jit
import numpy.typing as npt
from xgbsurv.models.utils import transform, transform_back


@jit(nopython=True)
def efron_likelihood(y: npt.NDArray[float], log_partial_hazard: npt.NDArray[float]) -> npt.NDArray[float]:
    """Generate negative loglikelihood (loss) according to Efron. Assumes times have been sorted beforehand.

    Parameters
    ----------
    y : npt.NDArray[float]
        Sorted array containing survival time and event where negative value is taken as censored event.
    log_partial_hazard : npt.NDArray[float]
        Estimated hazard.

    Returns
    -------
    npt.NDArray[float]
        Negative loglikelihood (loss) according to Efron.
    """
    # Assumes times have been sorted beforehand.
    time, event = transform_back(y)
    partial_hazard = np.exp(log_partial_hazard)
    samples = time.shape[0]
    previous_time = time[0]

    risk_set_sum = 0
    accumulated_sum = 0
    death_set_count = 0
    death_set_risk = 0
    likelihood = 0

    for i in range(samples):
        risk_set_sum += partial_hazard[i]

    for i in range(samples):
        sample_time = time[i]
        sample_event = event[i]
        sample_partial_hazard = partial_hazard[i]
        sample_partial_log_hazard = log_partial_hazard[i]

        if previous_time < sample_time:
            for ell in range(death_set_count):
                likelihood -= np.log(
                    risk_set_sum - ((ell / death_set_count) * death_set_risk)
                )
            risk_set_sum -= accumulated_sum
            accumulated_sum = 0
            death_set_count = 0
            death_set_risk = 0

        if sample_event:
            death_set_count += 1
            death_set_risk += sample_partial_hazard
            likelihood += sample_partial_log_hazard

        accumulated_sum += sample_partial_hazard
        previous_time = sample_time

    for ell in range(death_set_count):
        likelihood -= np.log(
            risk_set_sum - ((ell / death_set_count) * death_set_risk)
        )
    return -likelihood



@jit(nopython=True)
def calculate_sample_grad_hess_efron(
    sample_partial_hazard,
    sample_event,
    local_risk_set,
    local_risk_set_hessian,
    local_risk_set_death,
    local_risk_set_hessian_death,
):
    if sample_event:
        return (
            sample_partial_hazard * (local_risk_set_death)
        ) - sample_event, sample_partial_hazard * (local_risk_set_death) - (
            (local_risk_set_hessian_death)
        ) * (
            sample_partial_hazard**2
        )
    else:
        return (
            sample_partial_hazard * local_risk_set
        ), sample_partial_hazard * local_risk_set - local_risk_set_hessian * (
            sample_partial_hazard**2
        )


@jit(nopython=True)
def update_risk_sets_efron_pre(
    risk_set_sum,
    death_set_count,
    local_risk_set,
    local_risk_set_hessian,
    death_set_risk,
):
    local_risk_set_death = local_risk_set
    local_risk_set_hessian_death = local_risk_set_hessian

    for ell in range(death_set_count):
        contribution = ell / death_set_count
        local_risk_set += 1 / (risk_set_sum - (contribution) * death_set_risk)
        local_risk_set_death += (1 - (ell / death_set_count)) / (
            risk_set_sum - (contribution) * death_set_risk
        )
        local_risk_set_hessian += (
            1 / ((risk_set_sum - (contribution) * death_set_risk))
        ) ** 2

        local_risk_set_hessian_death += ((1 - contribution) ** 2) / (
            ((risk_set_sum - (contribution) * death_set_risk)) ** 2
        )

    return (
        local_risk_set,
        local_risk_set_hessian,
        local_risk_set_death,
        local_risk_set_hessian_death,
    )


@jit(nopython=True)
def efron_objective(y: npt.NDArray[float], log_partial_hazard: npt.NDArray[float]) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """Objective function calculating gradient and hessian of Efron negative likelihood function. Assumes data is sorted.

    Parameters
    ----------
    y : npt.NDArray[float]
        Sorted array containing survival time and event where negative value is taken as censored event.
    log_partial_hazard : npt.NDArray[float]
        Estimated hazard.

    Returns
    -------
    gradient : npt.NDArray[float]
        Gradient of the Efron negative likelihood (loss) function.
    hessian : npt.NDArray[float]]
        Diagonal Hessian of the Efron negative (loss) likelihood.
    """
    # Assumes times have been sorted beforehand.
    time, event = transform_back(y)
    partial_hazard = np.exp(log_partial_hazard)
    samples = time.shape[0]
    risk_set_sum = 0
    grad = np.empty(samples)
    hess = np.empty(samples)
    previous_time = time[0]
    local_risk_set = 0
    local_risk_set_hessian = 0
    death_set_count = 0
    censoring_set_count = 0
    accumulated_sum = 0
    death_set_risk = 0
    local_risk_set_death = 0
    local_risk_set_hessian_death = 0

    for i in range(samples):
        risk_set_sum += partial_hazard[i]

    for i in range(samples):
        sample_time = time[i]
        sample_event = event[i]
        sample_partial_hazard = partial_hazard[i]

        if previous_time < sample_time:
            if death_set_count:
                (
                    local_risk_set,
                    local_risk_set_hessian,
                    local_risk_set_death,
                    local_risk_set_hessian_death,
                ) = update_risk_sets_efron_pre(
                    risk_set_sum,
                    death_set_count,
                    local_risk_set,
                    local_risk_set_hessian,
                    death_set_risk,
                )
            for death in range(death_set_count + censoring_set_count):
                death_ix = i - 1 - death
                (
                    grad[death_ix],
                    hess[death_ix],
                ) = calculate_sample_grad_hess_efron(
                    partial_hazard[death_ix],
                    event[death_ix],
                    local_risk_set,
                    local_risk_set_hessian,
                    local_risk_set_death,
                    local_risk_set_hessian_death,
                )
            risk_set_sum -= accumulated_sum
            accumulated_sum = 0
            death_set_count = 0
            censoring_set_count = 0
            death_set_risk = 0
            local_risk_set_death = 0
            local_risk_set_hessian_death = 0

        if sample_event:
            death_set_count += 1
            death_set_risk += sample_partial_hazard
        else:
            censoring_set_count += 1

        accumulated_sum += sample_partial_hazard
        previous_time = sample_time

    i += 1
    if death_set_count:
        (
            local_risk_set,
            local_risk_set_hessian,
            local_risk_set_death,
            local_risk_set_hessian_death,
        ) = update_risk_sets_efron_pre(
            risk_set_sum,
            death_set_count,
            local_risk_set,
            local_risk_set_hessian,
            death_set_risk,
        )
    for death in range(death_set_count + censoring_set_count):
        death_ix = i - 1 - death
        (grad[death_ix], hess[death_ix],) = calculate_sample_grad_hess_efron(
            partial_hazard[death_ix],
            event[death_ix],
            local_risk_set,
            local_risk_set_hessian,
            local_risk_set_death,
            local_risk_set_hessian_death,
        )
    return grad, hess

@jit(nopython=True)
def efron_baseline_estimator(log_partial_hazard, time, event):

    # Assumes times have been sorted beforehand.
    n_unique_times = len(np.unique(time)) # check this part
    uniq_times = np.unique(time)
    partial_hazard = np.exp(log_partial_hazard)
    samples = time.shape[0]
    previous_time = time[0]
    risk_set_sum = 0
    accumulated_sum = 0
    death_set_count = 0
    death_set_risk = 0
    denominator = 0
    estimator = 0
    efron_estimator = np.zeros(n_unique_times)
    unique_event_times = np.zeros(n_unique_times)

    for i in range(samples):

        risk_set_sum += partial_hazard[i]
    q = 0

    for i in range(samples):

        sample_time = time[i]

        sample_event = event[i]

        sample_partial_hazard = partial_hazard[i]


        if previous_time < sample_time:

            for ell in range(death_set_count):

                denominator += risk_set_sum - (

                    (ell / death_set_count) * death_set_risk

                )

            if death_set_count:
                estimator += death_set_count / denominator
                efron_estimator[q] = estimator

                unique_event_times[q] = previous_time
                q += 1
            
            risk_set_sum -= accumulated_sum


            accumulated_sum = 0
            death_set_count = 0
            death_set_risk = 0
            denominator = 0



        if sample_event:
            death_set_count += 1
            death_set_risk += sample_partial_hazard



        accumulated_sum += sample_partial_hazard
        previous_time = sample_time

    for ell in range(death_set_count):
        
        denominator += risk_set_sum - (

            (ell / death_set_count) * death_set_risk

        )

    if death_set_count:
        estimator += death_set_count / denominator

        efron_estimator[q] += estimator

        unique_event_times[q] = previous_time

    cum_hazard_baseline = efron_estimator #double check this np.cumsum(efron_estimator)
    baseline_survival = np.exp(-cum_hazard_baseline)
    return uniq_times, cum_hazard_baseline, baseline_survival



class EfronPredictor(): 
    """Prediction functions particular to the Cox PH model"""
    
    def __init__(self) -> None:
        self.uniq_times = None
        self.cum_hazard_baseline = None
        self.baseline_survival = None
        
    
    def fit(self, partial_hazard, y):
        time, event = transform_back(y)
        self.uniq_times, self.cum_hazard_baseline, self.baseline_survival = efron_baseline_estimator(partial_hazard, time, event)
        print(self.uniq_times.shape, self.cum_hazard_baseline.shape)
        return self 

    def get_cumulative_hazard_function(self):
        return self.uniq_times, self.cum_hazard_baseline

    def get_survival_function(self):
        return self.uniq_times, self.baseline_survival

    def get_survival_function_own(self, partial_hazard):
        return self.uniq_times, np.exp(-self.cum_hazard_baseline)
