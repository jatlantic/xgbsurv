# Breslow final
from math import log
import numpy as np
import numpy.typing as npt
from numba import jit
#from scipy.special import logsumexp
import numpy.typing as npt
from xgbsurv.models.utils import transform, transform_back
import pandas as pd


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
    # Assumes times have been sorted beforehand.
    time, event = transform_back(y)
    if np.sum(event) == 0:
        raise RuntimeError("No events detected!")
    partial_hazard = np.exp(log_partial_hazard)
    n_events = np.sum(event)
    n_samples = time.shape[0]
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
    
    if set_count:
        likelihood -= set_count * log(risk_set_sum)
    
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

    # consider removing zero rows, would this be the right approach?
    risk_matrix = np.unique((np.outer(time,time)>=np.square(time)).astype(int).T, axis=0)
    denominator = np.sum(risk_score[None,:]*risk_matrix,axis=1)[::-1]     

    cum_hazard_baseline = np.cumsum(n_events / denominator)
    baseline_survival = np.exp(-cum_hazard_baseline)
    return uniq_times, cum_hazard_baseline, baseline_survival


# Looped version of breslow estimator

#@jit(nopython=True, cache=True, fastmath=True)
def breslow_estimator_loop(    
    predictor: np.array,
    time: np.array,
    event: np.array

):
    exp_predictor: np.array = np.exp(predictor)
    local_risk_set: float = np.sum(exp_predictor)
    event_mask: np.array = event.astype(np.bool_)
    n_unique_events: int = np.unique(time[event_mask]).shape[0]
    # unique event time does not work
    #cumulative_baseline_hazards: np.array = np.zeros(n_unique_events)
    # old modification cumulative_baseline_hazards: np.array = np.zeros(np.unique(time).shape)
    cumulative_baseline_hazards: np.array = np.zeros(n_unique_events)
    n_events_counted: int = 0
    local_death_set: int = 0
    accumulated_risk_set: float = 0
    previous_time: float = time[0]

    for _ in range(len(time)):
        sample_time: float = time[_]
        sample_event: int = event[_]
        sample_predictor: float = exp_predictor[_]

        if sample_time > previous_time and local_death_set:
            cumulative_baseline_hazards[n_events_counted] = local_death_set / (
                local_risk_set
            )

            local_death_set = 0
            local_risk_set -= accumulated_risk_set
            accumulated_risk_set = 0
            n_events_counted += 1

        if sample_event:
            local_death_set += 1
        accumulated_risk_set += sample_predictor
        previous_time = sample_time

    if local_death_set:
        cumulative_baseline_hazards[n_events_counted] = local_death_set / (
            local_risk_set
        )

    # old modification: cumulative_baseline_hazards[n_events_counted-1] = local_death_set / (
    #     local_risk_set
    # )
    # cumulative_baseline_hazards[n_events_counted] = local_death_set / (
    #     local_risk_set
    # )

    return (
        np.unique(time[event_mask]),
        np.cumsum(cumulative_baseline_hazards),
    )



def get_cumulative_hazard_function_breslow(X_train: np.array, 
        X_test: np.array, y_train: np.array, y_test: np.array,
        predictor_train: np.array, predictor_test: np.array
    ) -> pd.DataFrame:
    # inputs necessary: train_time, train_event, train_preds,
    # TODO: slim down inputs of function to what is needed
    for var in [X_train, X_train, y_train, y_train, predictor_train, predictor_test]:
        if not isinstance(var, np.ndarray):
                #print(type(var))
                var = var.values #to_numpy()
                #print(type(var))
    #print('y_train breslow final', y_train)
    time_train, event_train = transform_back(y_train)
    time_test, event_test = transform_back(y_test)
    time_test = np.unique(time_test)
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
    ) = breslow_estimator_loop(
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

# Breslow Predictor


class BreslowPredictor():
    """Prediction functions particular to the Cox PH model"""
    
    def __init__(self) -> None:
        self.uniq_times = None
        self.cum_hazard_baseline = None
        self.baseline_survival = None
        
    def get_cumulative_hazard_function(X_train: np.array, 
            X_test: np.array, y_train: np.array, y_test: np.array,
            predictor_train: np.array, predictor_test: np.array
        #self, X: np.array, time: np.array
        ) -> pd.DataFrame:
        # inputs necessary: train_time, train_event, train_preds, 
        time_train, event_train = transform_back(y_train)
        time_test, event_test = transform_back(y_test)
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
        ) = breslow_estimator_loop(
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
        cumulative_hazard_function: pd.DataFrame = pd.DataFrame(
            cumulative_baseline_hazards * np.exp(log_hazards),
            columns=time_test,
        )
        return cumulative_hazard_function

    def get_cumulative_hazard_function(self):
        return self.uniq_times, self.cum_hazard_baseline

    def get_survival_function(self):
        return self.uniq_times, self.baseline_survival

    def get_survival_function_own(self, partial_hazard):
        return self.uniq_times, np.exp(-self.cum_hazard_baseline)


# class BreslowPredictor(): # SET TIES OPTION
#     """Prediction functions particular to the Cox PH model"""
    
#     def __init__(self) -> None:
#         self.uniq_times = None
#         self.cum_hazard_baseline = None
#         self.baseline_survival = None
        
    
#     def fit(self, partial_hazard, y):
#         # CALL BRESLOW FUNCTIOIN FROM UTILS, no take the one below
#         time, event = transform_back(y)
#         self.uniq_times, self.cum_hazard_baseline, self.baseline_survival = breslow_estimator(partial_hazard, time, event)
#         print(self.uniq_times.shape, self.cum_hazard_baseline.shape)
#         return self 

#     def get_cumulative_hazard_function(self):
#         return self.uniq_times, self.cum_hazard_baseline

#     def get_survival_function(self):
#         return self.uniq_times, self.baseline_survival

#     def get_survival_function_own(self, partial_hazard):
#         return self.uniq_times, np.exp(-self.cum_hazard_baseline)





