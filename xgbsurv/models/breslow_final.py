# Breslow final
from math import log
import numpy as np
from numba import jit
#from scipy.special import logsumexp
import numpy.typing as npt
from xgbsurv.models.utils import transform, transform_back


# model functions
@jit(nopython=True)
def breslow_likelihood(y: npt.NDArray[float], log_partial_hazard: npt.NDArray[float]) -> npt.NDArray[float]:
    """Generate negative loglikelihood (loss) according to Breslow. 
    Assumes times have been sorted beforehand.

    Parameters
    ----------
    y : npt.NDArray[float]
        Sorted array containing survival time and event where negative value is taken as censored event.
    log_partial_hazard : npt.NDArray[float]
        Estimated hazard.

    Returns
    -------
    npt.NDArray[float]
        Negative loglikelihood (loss) according to Breslow.
    """
    time, event = transform_back(y)
    # Assumes times have been sorted beforehand.
    partial_hazard = np.exp(log_partial_hazard)
    n_events = np.sum(event)
    n_samples = time.shape[0]
    #print(n_samples)
    previous_time = time[0]
    risk_set_sum = 0
    likelihood = 0
    set_count = 0
    accumulated_sum = 0

    for i in range(n_samples):
        risk_set_sum += partial_hazard[i]

    for k in range(n_samples):
        current_time = time[k]
        if current_time > previous_time:
            # correct set-count, have to go back to set the different hazards for the ties
            likelihood -= set_count * log(risk_set_sum)
            risk_set_sum -= accumulated_sum
            set_count = 0
            accumulated_sum = 0

        if event[k]:
            set_count += 1
            likelihood += log_partial_hazard[k]

        previous_time = current_time
        accumulated_sum += partial_hazard[k]
    #print(likelihood)
    final_likelihood = -likelihood / n_events #n_samples
    return final_likelihood


@jit(nopython=True)
def update_risk_sets_breslow(
    risk_set_sum, death_set_count, local_risk_set, local_risk_set_hessian):
    local_risk_set += 1 / (risk_set_sum / death_set_count)
    local_risk_set_hessian += 1 / ((risk_set_sum**2) / death_set_count)
    return local_risk_set, local_risk_set_hessian


@jit(nopython=True)
def calculate_sample_grad_hess(
    sample_partial_hazard, sample_event, local_risk_set, local_risk_set_hessian
    ):
    return (
        sample_partial_hazard * local_risk_set
    ) - sample_event, sample_partial_hazard * local_risk_set - local_risk_set_hessian * (
        sample_partial_hazard**2
    )

@jit(nopython=True)
def breslow_objective(y: npt.NDArray[float], log_partial_hazard: npt.NDArray[float]) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """Objective function calculating gradient and hessian of Breslow negative likelihood function. Assumes data is sorted.

    Parameters
    ----------
    y : npt.NDArray[float]
        Sorted array containing survival time and event where negative value is taken as censored event.
    log_partial_hazard : npt.NDArray[float]
        Estimated hazard.

    Returns
    -------
    gradient : npt.NDArray[float]
        Gradient of the Breslow negative likelihood (loss) function.
    hessian : npt.NDArray[float]]
        Diagonal Hessian of the Breslow negative (loss) likelihood.
    """
    time, event = transform_back(y)
    # verify sorting, add printing error message
    #is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    # Assumes times have been sorted beforehand.
    #if is_sorted(time) == False:
    #    order = np.argsort(time, kind="mergesort")
    #    time = time[order]
    #   event = event[order]
    #    log_partial_hazard = log_partial_hazard[order]



    partial_hazard = np.exp(log_partial_hazard)
    samples = time.shape[0]
    risk_set_sum = 0

    for i in range(samples):
        risk_set_sum += partial_hazard[i]

    grad = np.empty(samples)
    hess = np.empty(samples)
    previous_time = time[0]
    local_risk_set = 0
    local_risk_set_hessian = 0
    death_set_count = 0
    censoring_set_count = 0
    accumulated_sum = 0

    for i in range(samples):
        sample_time = time[i]
        sample_event = event[i]
        sample_partial_hazard = partial_hazard[i]


        if previous_time < sample_time:
            if death_set_count:
                (
                    local_risk_set,
                    local_risk_set_hessian,
                ) = update_risk_sets_breslow(
                    risk_set_sum,
                    death_set_count,
                    local_risk_set,
                    local_risk_set_hessian,
                )

            for death in range(death_set_count + censoring_set_count):
                death_ix = i - 1 - death
                (grad[death_ix], hess[death_ix],) = calculate_sample_grad_hess(
                    partial_hazard[death_ix],
                    event[death_ix],
                    local_risk_set,
                    local_risk_set_hessian,
                )


            risk_set_sum -= accumulated_sum
            accumulated_sum = 0
            death_set_count = 0
            censoring_set_count = 0



        if sample_event:
            death_set_count += 1
        else:
            censoring_set_count += 1

        accumulated_sum += sample_partial_hazard
        previous_time = sample_time

    i += 1
    if death_set_count:
        local_risk_set, local_risk_set_hessian = update_risk_sets_breslow(
            risk_set_sum,
            death_set_count,
            local_risk_set,
            local_risk_set_hessian,
        )

    for death in range(death_set_count + censoring_set_count):
        death_ix = i - 1 - death
        (grad[death_ix], hess[death_ix],) = calculate_sample_grad_hess(
            partial_hazard[death_ix],
            event[death_ix],
            local_risk_set,
            local_risk_set_hessian,
        )
        #print('grad',-grad/samples)
        #print('hess',hess)
    #print('grad, hess',grad,hess)
    return grad, hess #- /samples - the minus creates wrong results in xgboost, /samples seems to lower concordance index
    # divide by number of events instead (ie sum of events==1)


# estimator
def breslow_estimator(log_hazard, time, event):
    #time, event = transform_back(y)
    risk_score = np.exp(log_hazard)

    is_sorted = lambda a: np.all(a[:-1] <= a[1:])

    if is_sorted(time) == False:
        order = np.argsort(time, kind="mergesort")
        time = time[order]
        event = event[order]
        risk_score = risk_score[order]

    uniq_times = np.unique(time)
    idx = np.digitize(time, np.unique(time))
    breaks = np.flatnonzero(np.concatenate(([1], np.diff(idx))))
    # numpy diff nth discrete difference over index, add 1 at the beginning
    # flatnonzero return indices that are nonzero in flattened version
    n_events = np.add.reduceat(event, breaks, axis=0)

    # consider removing zero rows
    risk_matrix = np.unique((np.outer(time,time)>=np.square(time)).astype(int).T, axis=0)
    denominator = np.sum(risk_score[None,:]*risk_matrix,axis=1)[::-1]     

    cum_hazard_baseline = np.cumsum(n_events / denominator)
    baseline_survival = np.exp(-cum_hazard_baseline)
    return uniq_times, cum_hazard_baseline, baseline_survival


# Breslow Predictor


class BreslowPredictor(): # SET TIES OPTION
    """Prediction functions particular to the Cox PH model"""
    
    def __init__(self) -> None:
        self.uniq_times = None
        self.cum_hazard_baseline = None
        self.baseline_survival = None
        
    
    def fit(self, partial_hazard, y):
        # CALL BRESLOW FUNCTIOIN FROM UTILS, no take the one below
        time, event = transform_back(y)
        self.uniq_times, self.cum_hazard_baseline, self.baseline_survival = breslow_estimator(partial_hazard, time, event)
        print(self.uniq_times.shape, self.cum_hazard_baseline.shape)
        return self 

    def get_cumulative_hazard_function(self):
        return self.uniq_times, self.cum_hazard_baseline

    def get_survival_function(self):
        return self.uniq_times, self.baseline_survival

    def get_survival_function_own(self, partial_hazard):
        return self.uniq_times, np.exp(-self.cum_hazard_baseline)





