# Efron final
from math import log # change below
import numpy as np
from numba import jit
import numpy.typing as npt
from xgbsurv.models.utils import transform, transform_back
import pandas as pd


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


@jit(nopython=True, cache=True, fastmath=True)
def efron_estimator(
    predictor: np.array,
    time: np.array,
    event: np.array,
):
    exp_predictor: np.array = np.exp(predictor)
    local_risk_set: float = np.sum(exp_predictor)
    event_mask: np.array = event.astype(np.bool_)
    n_unique_events: int = np.unique(time[event_mask]).shape[0]
    cumulative_baseline_hazards: np.array = np.zeros(n_unique_events)
    n_events_counted: int = 0
    local_death_set: int = 0
    accumulated_risk_set: float = 0
    previous_time: float = time[0]
    local_death_set_risk: float = 0

    for _ in range(len(time)):
        sample_time: float = time[_]
        sample_event: int = event[_]
        sample_exp_predictor: float = exp_predictor[_]

        if sample_time > previous_time and local_death_set:
            for ell in range(local_death_set):
                cumulative_baseline_hazards[n_events_counted] += 1 / (
                    local_risk_set
                    - (ell / local_death_set) * local_death_set_risk
                )

            local_risk_set -= accumulated_risk_set
            accumulated_risk_set = 0
            local_death_set_risk = 0
            local_death_set = 0
            n_events_counted += 1

        if sample_event:
            local_death_set += 1
            local_death_set_risk += sample_exp_predictor
        accumulated_risk_set += sample_exp_predictor
        previous_time = sample_time

    for ell in range(local_death_set):
        cumulative_baseline_hazards[n_events_counted] += 1 / (
            local_risk_set - (ell / local_death_set) * local_death_set_risk
        )

    return (
        np.unique(time[event_mask]),
        np.cumsum(cumulative_baseline_hazards),
    )


def get_cumulative_hazard_function_efron(X_train: np.array, 
        X_test: np.array, y_train: np.array, y_test: np.array,
        predictor_train: np.array, predictor_test: np.array
    ) -> pd.DataFrame:
    # inputs necessary: train_time, train_event, train_preds, 
    time_train, event_train = transform_back(y_train)
    time_test, event_test = transform_back(y_test)
    #print(time_test)
    if np.min(time_test) < 0:
        raise ValueError(
            "Times for survival and cumulative hazard prediction must be greater than or equal to zero."
            + f"Minimum time found was {np.min(time_test)}."
            + "Please remove any times strictly less than zero."
        )
    cumulative_baseline_hazards_times: np.array
    cumulative_baseline_hazards: np.array
    (
        cumulative_baseline_hazards_times,
        cumulative_baseline_hazards,
    ) = efron_estimator(
        time=time_train, event=event_train, predictor=predictor_train
    )
    cumulative_baseline_hazards = np.concatenate(
        [np.array([0.0]), cumulative_baseline_hazards]
    )
    cumulative_baseline_hazards_times: np.array = np.concatenate(
        [np.array([0.0]), cumulative_baseline_hazards_times]
    )
    cumulative_baseline_hazards: np.array = np.tile(
        A=cumulative_baseline_hazards[
            np.digitize(
                x=time_test, bins=cumulative_baseline_hazards_times, right=False
            )
            - 1
        ],
        reps=X_test.shape[0],
    ).reshape((X_test.shape[0], time_test.shape[0]))
    log_hazards: np.array = (
        np.tile(
            A= predictor_test, #self.predict(X),
            reps=time_test.shape[0],
        )
        .reshape((time_test.shape[0], X_test.shape[0]))
        .T
    )
    df_cumulative_hazard: pd.DataFrame = pd.DataFrame(
        cumulative_baseline_hazards * np.exp(log_hazards),
        columns=time_test,
    )

    return df_cumulative_hazard.T.sort_index(axis=1)






















#@jit(nopython=True) np.insert and numba not working
def efron_baseline_estimator(log_partial_hazard, time, event):

    # currently censoreed events disappear from the estimate
    # need to deal with that!!!


    # Assumes times have been sorted beforehand.
    n_unique_times = len(np.unique(time[event==1])) # check this part
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
    efron_estimator = np.unique(time[event==1]) #np.zeros(n_unique_times)
    unique_event_times = np.unique(time[event==1]) #np.zeros(n_unique_times)

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

    # estimator += (death_set_count * unique_death_times) / denominator
    # estimator = np.cumsum(estimator)
    #print('q value at the end', q)
    # If for a certain time the event value is always zero it is not captured
    # In that case we add that value

    # TODO: Adapt unique_event_times for the time that might be missing
    # time does not seem to be sorted anymore
    cum_hazard_baseline_final = efron_estimator.copy()
    #print('shape cum_hazard_baseline_final', cum_hazard_baseline_final.shape)
    for t in uniq_times:
        if t not in unique_event_times:
            thres = t
            print('thres', thres)
            try:
                ind = np.argmax(unique_event_times[unique_event_times <= thres])
            except:
                ind = 0
            #print('ind', ind)
            #print('shape cum_hazard_baseline_final', cum_hazard_baseline_final.shape)
            val = cum_hazard_baseline_final[ind]
            cum_hazard_baseline_final = np.insert(cum_hazard_baseline_final, ind, val)
    #cum_hazard_baseline = efron_estimator #double check this np.cumsum(efron_estimator)
    baseline_survival = np.exp(-cum_hazard_baseline_final) #verify if this is not repeated
    #print(len(uniq_times), len(unique_event_times), len(cum_hazard_baseline_final), len(baseline_survival))
    return uniq_times, cum_hazard_baseline_final, baseline_survival



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
