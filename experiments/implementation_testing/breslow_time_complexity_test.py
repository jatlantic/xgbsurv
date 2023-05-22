import numpy as np
import pandas as pd
from xgbsurv.models.utils import transform
from xgbsurv.models.breslow_final import breslow_likelihood, breslow_objective, transform_back
from scipy.special import logsumexp
import jax.numpy as jnp
from jax import  grad, hessian #jit,
import jax.scipy.special as jsp
import time as t
from math import log
import numba
from numba import jit #as njit

# vectorized coxph

def get_risk_matrix(time):
    return (np.outer(time, time) >= np.square(time)).astype(int).T

def cox_ph_loss(log_partial_hazard, time, event):
    #time, event = transform_back(y)
    risk_matrix = get_risk_matrix(time)
    hazard_risk = log_partial_hazard*risk_matrix
    inp = event*(log_partial_hazard - logsumexp(hazard_risk, b=risk_matrix, axis=1))
    # logsumexp numerically more stable than numpy
    loss = np.sum(inp)
    return -loss/event.sum()

# sksurv approach

def cox_ph_loss_sksurv(y_pred, time, event):
    # sksurv approach without cython
    #time, event = transform_back(y)
    n_samples = event.shape[0]
    loss = 0

    for i in range(n_samples):
        at_risk = 0
        for j in range(n_samples):
            if time[j] >= time[i]:
                at_risk += np.exp(y_pred[j])
        loss += event[i] * (y_pred[i] - np.log(at_risk))
    return - loss/event.sum()

def breslow_likelihood(log_partial_hazard, time, event):

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

@jit(nopython=True, cache=True, fastmath=True)
def breslow_likelihood_numba(log_partial_hazard, time, event):

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


## Run comparison
def function1(hazard,time, event):
    return cox_ph_loss(hazard, time, event)

def function2(hazard,time, event):
    return breslow_likelihood(hazard,time, event)   

def function3(hazard,time, event):
    return breslow_likelihood_numba(hazard, time, event)

path = '/Users/JUSC/Documents/xgbsurv_benchmarking/implementation_testing/simulation_data'
def comparison(num_runs = 10, size=1000):
    hazard = log_hazard = np.random.normal(0, 1, size)
    df = pd.read_csv(path+'/survival_simulation_'+str(size)+'.csv')
    df.sort_values(by='time', inplace=True)
    time = df.time.to_numpy()
    event = df.event.to_numpy()
    # Empty list to store the execution times
    function1_times = []
    function2_times = []
    function3_times = []

    # Loop to run each function and record the execution times
    for i in range(num_runs):
        #print('Running Function 1')
        start_time = t.time()
        function1(hazard,time, event)
        end_time = t.time()
        function1_times.append(end_time - start_time)

        #print('Running Function 2')
        start_time = t.time()
        function2(hazard,time, event)
        end_time = t.time()
        function2_times.append(end_time - start_time)

        #print('Running Function 3')
        start_time = t.time()
        function3(hazard,time, event)
        end_time = t.time()
        function3_times.append(end_time - start_time)

    # Calculate the mean and standard deviation of the execution times for each function
    function1_mean = sum(function1_times) / len(function1_times)
    function1_std = pd.Series(function1_times).std()
    function2_mean = sum(function2_times) / len(function2_times)
    function2_std = pd.Series(function2_times).std()
    function3_mean = sum(function3_times) / len(function3_times)
    function3_std = pd.Series(function3_times).std()

    # Create a Pandas dataframe to display the results
    df = pd.DataFrame({
        'Function': ['Standard Vectorized CoxPH', 'Breslow', 'Breslow Numba'],
        'Mean': [function1_mean, function2_mean, function3_mean],
        'Standard Deviation': [function1_std, function2_std, function3_std],
        'Sample Size': [size, size, size],
        'Number Repetitions': [num_runs, num_runs, num_runs]
    })
    return df

df_1000 = comparison(num_runs = 100, size=1000)
df_10000 = comparison(num_runs = 100, size=10000)
#df_100000 = comparison(num_runs = 100, size=100000)

dff = pd.concat([df_1000,df_10000]) #, df_100000
dff.to_csv(path+'/results/breslow_numba_comparison.csv', index=False)
print(dff.to_latex(index=False))