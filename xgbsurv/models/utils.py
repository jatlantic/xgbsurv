from numba import jit
import numpy as np
from scipy.special import softmax
#from lifelines.datasets import load_rossi
from sklearn.model_selection import train_test_split
import pandas as pd
# using numpy.typing.NDArray[A] as an alias for numpy.ndarray[Any, numpy.dtype[A]]:
# discuss adding dimensionality to it
# https://stackoverflow.com/questions/71109838/numpy-typing-with-specific-shape-and-datatype
import numpy.typing as npt


@jit(nopython=True, cache=True, fastmath=True)
def transform(time: npt.NDArray[float], event: npt.NDArray[int]) -> npt.NDArray[float]:
    """Transforms time, event into XGBoost digestable format.

    Parameters
    ----------
    time : npt.NDArray[float]
        Survival time.
    event : npt.NDArray[int]
        Boolean event indicator. Zero value is taken as censored event.

    Returns
    -------
    y : npt.NDArray[float]
        Transformed array containing survival time and event where negative value is taken as censored event.
    """
    #if isinstance(time, pd.Series):
    #    time = time.to_numpy()
    #    event = event.to_numpy()
    event_mod = np.copy(event) 
    event_mod[event_mod==0] = -1
    if np.any(time==0):
        raise RuntimeError('Data contains zero time value!')
        # alternative: time[time==0] = np.finfo(float).eps
    y = event_mod*time
    return y



@jit(nopython=True, cache=True, fastmath=True) # not really needed
def transform_back(y: npt.NDArray[float]) -> tuple[npt.NDArray[float], npt.NDArray[int]]:
    """Transforms XGBoost digestable format variable y into time and event.

    Parameters
    ----------
    y : npt.NDArray[float]
        Array containing survival time and event where negative value is taken as censored event.

    Returns
    -------
    tuple[npt.NDArray[float],npt.NDArray[int]]
        Survival time and event.
    """
    time = np.abs(y)
    event = (np.abs(y) == y)
    event = event.astype(np.int64) # for numba
    return time, event

@jit(nopython=True)
def sort_X_y(X, y):
    # naming convention as in sklearn
    # add sorting here, maybe there is a faster way
    y_abs = np.absolute(y)
    if np.all(np.diff(y_abs) >= 0) is False:
        #print('Values are being sorted!')
        order = np.argsort(y_abs, kind="mergesort")
        y = y[order]
        X = X[order]
    return X, y
    



# change to small letter in the end to keep convention
def KaplanMeier(time: npt.NDArray[float], event: npt.NDArray[int], 
                cens_dist: bool = False
) -> tuple[npt.NDArray[float], npt.NDArray[float]] | tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[int]]:
    """_summary_

    Parameters
    ----------
    time : npt.NDArray[float]
        _description_
    event : npt.NDArray[int]
        _description_
    cens_dist : bool, optional
        _description_, by default False

    Returns
    -------
    tuple[npt.NDArray[float], npt.NDArray[float]] | tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[int]]
        _description_
    
    References
    ----------
    .. [1] Kaplan, E. L. and Meier, P., "Nonparametric estimation from incomplete observations",
           Journal of The American Statistical Association, vol. 53, pp. 457-481, 1958.
    .. [2] S. Pölsterl, “scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn,”
           Journal of Machine Learning Research, vol. 21, no. 212, pp. 1–6, 2020.
    """
    # similar approach to sksurv, but no loops
    # even and censored is other way round in sksurv ->clarify
    #time, event = transform_back(y)
    # order, remove later
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    
    if is_sorted(time) == False:
        order = np.argsort(time, kind="mergesort")
        time = time[order]
        event = event[order]
    
    times = np.unique(time)
    idx = np.digitize(time, np.unique(time))
    # numpy diff nth discrete difference over index, add 1 at the beginning
    breaks = np.flatnonzero(np.concatenate(([1], np.diff(idx))))

    # flatnonzero return indices that are nonzero in flattened version
    n_events = np.add.reduceat(event, breaks, axis=0)
    n_at_risk = np.sum(np.unique((np.outer(time,time)>=np.square(time)).astype(int).T,axis=0),axis=1)[::-1]
    
    # censoring distribution for ipcw estimation
    #n_censored a vector, with 1 at censoring position, zero elsewhere
    if cens_dist:
        n_at_risk -= n_events
        # for each unique time step how many observations are censored
        censored = 1-event
        n_censored = np.add.reduceat(censored, breaks, axis=0)
        vals = 1-np.divide(
        n_censored, n_at_risk,
        out=np.zeros(times.shape[0], dtype=float),
        where=n_censored != 0,
    )
        
        estimates = np.cumprod(vals)
        return times, estimates, n_censored


    else:
        vals = 1-np.divide(
        n_events, n_at_risk,
        out=np.zeros(times.shape[0], dtype=float),
        where=n_events != 0,
        )
        estimates = np.cumprod(vals)
        return times, estimates

def ipcw_estimate(time: npt.NDArray[float], event: npt.NDArray[int]) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """IPCW

    Parameters
    ----------
    time : npt.NDArray[float]
        _description_
    event : npt.NDArray[int]
        _description_

    Returns
    -------
    tuple[npt.NDArray[float], npt.NDArray[float]]
        _description_
    """
    unique_time, cens_dist, n_censored = KaplanMeier(time, event, cens_dist=True) 
    #print(cens_dist)
    # similar approach to sksurv
    idx = np.searchsorted(unique_time, time)
    est = 1.0/cens_dist[idx] # improve as divide by zero
    est[n_censored[idx]!=0] = 0
    # in R mboost there is a maxweight of 5
    est[est>5] = 5
    return unique_time, est



# create concordance matrix for Deephit loss

def concordance_mat(idx_durations, event):
    # create concordance matrix for Deephit loss
    # follows Pycox approach
    n = len(idx_durations)
    mat = np.zeros((n, n))
    for i in range(n):
        dur_i = idx_durations[i]
        ev_i = event[i]
        if ev_i == 0:
            continue
        for j in range(n):
            dur_j = idx_durations[j]
            ev_j = event[j]
            if (dur_i < dur_j) or ((dur_i == dur_j) and (ev_j == 0)):
                mat[i, j] = 1
    return mat






# helper functions

def get_risk_matrix(time:np.ndarray):
    return (np.outer(time, time) >= np.square(time)).astype(int).T

def get_death_matrix(time, event):
    return (np.outer(time, time) == np.square(time * event)).astype(int).T

def get_risk_matrix_efron(time):
    return (np.outer(time, time) > np.square(time)).astype(int).T




def make_dataset(type='discrete'):
    """Create dataset for testing."""
    types = ['discrete', 'continuous', 'multi-discrete']
    if type not in types:
        raise ValueError("Invalid type. Expected one of: %s" % types)
    if type == 'discrete':
        time = np.round(np.random.uniform(1, 100, 100)).astype('int64')
        event = np.random.binomial(1, 0.5, 100)
        log_hazard = np.random.normal(0, 1, 100)
    if type == 'continuous':
        time = np.random.uniform(1, 100, 100)
        event = np.random.binomial(1, 0.5, 100).astype('bool')
        log_hazard = np.random.normal(0, 1, 100)
    if type == 'multi-discrete':
        event = np.random.binomial(1, 0.5, 100).astype('bool')
        kclasses = event.shape[0]
        log_hazard = np.random.lognormal(0.5, 0.1, (100, kclasses)).astype('float64') # to obtain positive values
        time = np.round(np.random.uniform(1, 100, 100)).astype('int64')
        # sort
        order = time.argsort(axis=0)
        time = time[order]
        log_hazard = log_hazard[order]
        event = event[order]
        log_hazard = softmax(log_hazard, axis=0)-0.001 # this has to be discussed
    return time, event, log_hazard

#time, event, log_hazard = make_dataset(type='multi-discrete')

def make_xgb_dataset(type='test-xgb'):
    types = ['test-xgb']
    if type not in types:
        raise ValueError("Invalid type. Expected one of: %s" % types)
    if type == 'test-xgb':
        rossi = load_rossi()
        y = transform(rossi.week.to_numpy(), rossi.arrest.to_numpy())
        X = rossi[['fin', 'age', 'race', 'wexp', 'mar', 'paro', 'prio']].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    return X_train, X_test, y_train, y_test

    




# Deephit
# this is a simplified equidistant discretizer
# a more sophisticated version can be found in pycox

def discretizer_df(df, n_cuts=10, type = 'equidistant', min_time=0.0) -> pd.DataFrame:
    """Discretize dataframe along time axis.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    n_cuts : int, optional
        _description_, by default 10
    type : str, optional
        _description_, by default 'equidistant'
    min_time : float, optional
        _description_, by default 0.0

    Returns
    -------
    pd.DataFrame
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    
    References
    ----------
    .. [1] Kvamme, H. havakv/pycox. (2023).
    .. [2] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """
    if 'time' and 'event' not in df.columns:
        raise ValueError('Required columns are not in dataframe.')
    elif df.time.min()==0:
        raise ValueError('Time zero cannot exist in the data.')
    elif type=='equidistant':
        df = df.sort_values(by='time', ascending=True)
        time, _ = df.time.to_numpy(), df.event.to_numpy()
        cuts = np.linspace(min_time,time.max(),num=n_cuts)
        idx = np.digitize(time, cuts, right=False) #-1
        df['time'] = idx
    elif type=='quantiles':
        df = df.sort_values(by='time', ascending=True)
        time, _ = df.time.to_numpy(), df.event.to_numpy()
        surv_durations, surv_est = KaplanMeier(df.time.to_numpy(), df.event.to_numpy())
        # this is like in pycox, see citation above
        preliminary_cuts = np.linspace(surv_est.min(), surv_est.max(), n_cuts)
        cuts_index = np.searchsorted(surv_est[::-1], preliminary_cuts)[::-1]
        final_cuts = np.unique(surv_durations[::-1][cuts_index])
        if len(final_cuts) != n_cuts:
            print(f"{len(final_cuts)} cuts are used instead of {n_cuts} since original ones are not unique.")
        # until here
        # +1 as zero time steps should not exist
        idx = np.digitize(time, final_cuts, right=False)+1

        df['time'] = idx
    return df



def breslow_estimator(log_hazard, y):
    # final breslow estimator
    time, event = transform_back(y)
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


# use arange with reduceat to account for ties in breslow

def breslow_estimator_j(y: np.array,log_hazard: np.array, tie_correction="efron"):
    time, event = transform_back(y)
    risk_score = np.exp(log_hazard)


    is_sorted = lambda a: np.all(a[:-1] <= a[1:])

    if is_sorted(time) == False:
        order = np.argsort(time, kind="mergesort")
        time = time[order]
        event = event[order]
        risk_score = risk_score[order]

    unique_times, ind, counts = np.unique(time, return_index=True, return_counts=True)
    print(unique_times, ind, counts)
    
    if tie_correction=="breslow":

        # correct this for easier approach
        idx = np.digitize(time, np.unique(time))
        breaks = np.flatnonzero(np.concatenate(([1], np.diff(idx))))
        # numpy diff nth discrete difference over index, add 1 at the beginning
        # flatnonzero return indices that are nonzero in flattened version
        n_events = np.add.reduceat(event, breaks, axis=0)

        # consider removing zero rows
        risk_matrix = np.unique((np.outer(time,time)>=np.square(time)).astype(int).T, axis=0)
        denominator = np.sum(risk_score[None,:]*risk_matrix,axis=1)[::-1]     

        baseline_cum_hazard = np.cumsum(n_events / denominator)
        return unique_times, baseline_cum_hazard
    
    if tie_correction=="efron":
        
        n = event.shape[0]
        tie_rep = np.repeat(n-ind,counts)
        adapt_ind = np.concatenate([np.arange(i) for i in (counts) ])
        print(adapt_ind)
        tie_rep_efron = tie_rep - adapt_ind
        print(tie_rep)
        breaks = np.array([np.zeros(n), tie_rep_efron]).T.flatten()
        breaks = breaks.astype('int64')
        # reverse order
        risk_score = risk_score[::-1]
        # add zero because closed brackets behaviour of reduceat
        to_sum = np.append(risk_score,0)

        denominator = np.add.reduceat(to_sum, breaks)[::2]
        cnt_vector = np.clip(event, 0, 2)
        numerator = np.cumsum(cnt_vector)
        # sum num and denom for unique timesteps
        breaks2 = np.array([ind, ind+counts]).T.flatten()
        numerator = np.append(numerator,0)
        denominator = np.append(denominator,0)
        numerator = np.add.reduceat(numerator,breaks2 )[::2]
        denominator = np.add.reduceat(denominator,breaks2 )[::2]
        baseline_cum_hazard = np.cumsum(numerator / denominator)
        return unique_times, baseline_cum_hazard



#breslow_estimator_j(y,log_hazard, tie_correction="breslow")
#breslow_estimator_j(y,log_hazard, tie_correction="efron")

def breslow_estimator_new(
    log_hazard: np.array, y: np.array, tie_correction="efron"
):
    #beta: np.array = check_array(beta)
    #X: np.array = check_array(X)
    #check_y_survival(y)
    #check_cox_tie_correction(tie_correction)

    #time: np.array
    #event: np.array
    print(y)
    time, event = transform_back(y)
    print(time)
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    partial_hazards = np.exp(log_hazard)
    if is_sorted(time) == False:
        order = np.argsort(time, kind="mergesort")
        time = time[order]
        event = event[order]
        partial_hazards = partial_hazards[order]
    print(time)
    risk_matrix = get_risk_matrix(time)
    
    
    #partial_hazards = np.exp(np.dot(beta, X))
    if tie_correction == "efron":
        log_partial_hazard = log_hazard
        _, ind, counts = np.unique(time, return_index=True, return_counts=True)
        print('counts',counts)
        unique_times = time[np.sort(ind)]
        unique_death_times = []
        # this could be simplified with unique I believe
        for ix, t in enumerate(unique_times):
            selected_ix = np.where(time == t)[0]
            selected_ix = selected_ix[np.where(event[selected_ix] == 1)[0]]
            #print('selected_ix', selected_ix)
            if not np.any(event[selected_ix]):
                continue
            else:
                unique_death_times += [selected_ix[0]]
        # gives rows of unique death times where event==1
        #print(unique_death_times)
        unique_death_times = np.array(unique_death_times).astype(int)
        #print('unique_death_times',unique_death_times)
        death_matrix = get_death_matrix(time, event)
        # subset death matrix for unique event==1 death times
        death_matrix = death_matrix[:, unique_death_times]
        risk_matrix = get_risk_matrix(time)
        # subset risk matrix for unique event==1 death times
        risk_matrix = risk_matrix[unique_death_times, :]
        # sum death matrix along rows
        death_matrix_sum = np.sum(death_matrix, axis=0)
        # fill with exp(hazard)
        # death_matrix here is matrix with deaths per unique time step column wise
        # fill with partial hazards
        death_set_partial_hazard = np.matmul(
            np.exp(log_partial_hazard), death_matrix
        )
        # fill risk set with partial hazards
        risk_set_partial_hazard = np.matmul(
            np.exp(log_partial_hazard), risk_matrix.T
        )
        # np.expand_dims ->Insert a new axis that will appear at the axis position in the expanded array shape.
        # create js formula where each column contains the range per unique time step
        # unique death times are only where an event happened, ie death not censoring
        print('expdims', np.expand_dims(np.arange(np.max(death_matrix_sum)), 1))
        efron_matrix = np.repeat(
            np.expand_dims(np.arange(np.max(death_matrix_sum)), 1),
            repeats=death_matrix_sum.shape[0],
            axis=1,
        )
        print('efron matrix',efron_matrix)
        # create helper matrix that has the same size as efron matrix
        helper_matrix = np.zeros(efron_matrix.shape)
        # go over each position and value of final death number (vector) in unique time step
        #print('death_matrix_sum', death_matrix_sum)
        # and set the rows of efron matrix from death number onwards to zero
        # at column of death matrix value vector using index
        # do the same for helper matrix which has the same shape as efron matrix
        # and instead of zero set to 1
        for ix, qx in enumerate(death_matrix_sum):
            print('qx',qx)
            efron_matrix[qx:, ix] = 0 # basically set to zero if range to wide
            helper_matrix[qx:, ix] = 1 # set to 1 where range is too wide
        # ie. this means we have one as the factor for the hazards

        risk_set_sums = np.prod(
            risk_set_partial_hazard
            - risk_set_partial_hazard * helper_matrix
            - (efron_matrix / death_matrix_sum) * death_set_partial_hazard
            + helper_matrix, axis=0) # this one does not seem to make a difference,
            #not sure of its role, but the value(s) is used for matmul later
            # but potentially it helps against nan and inf values
        
    #     risk_matrix_efron = get_risk_matrix_efron(time)
    #     risk_set_sums = np.prod(
    #     np.matmul(
    #         risk_set_sums0,
    #         risk_matrix_efron[unique_death_times, :]
    #         * np.exp(log_partial_hazard),
    #     ),
    #     axis=0,
    # )
        #risk_set_sums_rep = 
        print('death_matrix_sum',get_death_matrix(time, event).shape)
        print('risk_set_sums',risk_set_sums.shape)
    elif tie_correction == "breslow":
        #dtype('int64')
        risk_set_sums = np.sum(
            partial_hazards.repeat(time)
            .reshape((partial_hazards.shape[0], time.shape[0]))
            .T
            * risk_matrix,
            axis=0,
        )
    print('risksetsums',risk_set_sums)
    ix: np.array = np.argsort(time)
    print('ix', ix)
    print('time', time)
    sorted_time: np.array = time[ix]
    print('sorted_time',sorted_time)
    sorted_event: np.array = event[ix].astype('bool')
    #sorted_risk_set_sums: np.array = risk_set_sums[ix]
    sorted_risk_set_sums = np.repeat(risk_set_sums, counts)
    # TODO: Get rid of list comprehension
    # np.array() before
    unique_event_times_ix = np.concatenate([np.where(sorted_time[event] == time)[0] for time in sorted_time])
    print('unique_event_times_ix', unique_event_times_ix)
    print('next')
    death_counts = np.unique(sorted_time[event], return_counts=True)
    print('sorted_time[event]',sorted_time[event])
    print('death_counts',death_counts[1])
    return (
        np.cumsum(death_counts[1] / sorted_risk_set_sums[sorted_event][unique_event_times_ix]),
        sorted_time[sorted_event][unique_event_times_ix],
    )


