
import torch
from torch.nn.modules.loss import _Loss
from math import log
import numpy as np
import math
from math import exp, sqrt, pi, erf, pow
import pandas as pd


# torch transform function

def transform_torch(time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
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
    event_mod = event.clone()
    event_mod[event_mod==0] = -1
    if (time==0).any():
        raise RuntimeError('Data contains zero time value!')
        # alternative: time[time==0] = np.finfo(float).eps
    y = event_mod*time
    return y.to(torch.float32)


def transform_back_torch(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    time = torch.abs(y)
    event = (torch.abs(y) == y)
    event = event # for numba
    return time.to(torch.float32), event.to(torch.float32)

def transform_back_torch_deephit(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    #TODO: Build conditions and combine transform functions
    y = y[:,0]
    time = torch.abs(y)
    event = (torch.abs(y) == y)
    event = event # for numba
    return time.to(torch.float32), event.to(torch.float32)


# Breslow  loss

def breslow_likelihood_torch(y: torch.Tensor, log_partial_hazard: torch.Tensor) -> torch.Tensor:
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
    
    #print('log partial hazard', log_partial_hazard)
    if isinstance(log_partial_hazard, np.ndarray):
        log_partial_hazard = torch.from_numpy(log_partial_hazard)
    if isinstance(log_partial_hazard, pd.Series):
        log_partial_hazard = torch.tensor(log_partial_hazard.values)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    if isinstance(y, pd.Series):
        y = torch.tensor(y.values)

    #print(type(y))
    #print(y)
    time, event = transform_back_torch(y)
    # Assumes times have been sorted beforehand.
    partial_hazard = torch.exp(log_partial_hazard)
    #print('partial hazard', partial_hazard)
    n_events = torch.sum(event)
    n_samples = time.shape[0]
    #print(n_samples)
    previous_time = time[0]
    risk_set_sum = 0
    likelihood = 0
    set_count = 0
    accumulated_sum = 0
    final_likelihood = 0

    for i in range(n_samples):
        risk_set_sum = risk_set_sum+partial_hazard[i]

    for k in range(n_samples):
        current_time = time[k]
        if current_time > previous_time:
            # correct set-count, have to go back to set the different hazards for the ties
            likelihood = likelihood -(set_count * torch.log(risk_set_sum))
            risk_set_sum = risk_set_sum - accumulated_sum
            set_count = 0
            accumulated_sum = 0

        if event[k]:
            set_count = set_count + 1
            likelihood = likelihood + log_partial_hazard[k]

        previous_time = current_time
        accumulated_sum = accumulated_sum+ partial_hazard[k]
    #print('likelihood',likelihood)
    final_likelihood = final_likelihood-(likelihood / n_events) #n_samples
    return final_likelihood #/n_samples

class BreslowLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
            super().__init__(size_average, reduce, reduction)
            # Initialize any additional variables you need for your custom loss function here.

    def forward(self, prediction, input):
        loss = breslow_likelihood_torch(input, prediction)
        return loss.to(torch.float32)
    

# Efron  loss

def efron_likelihood_torch(y: torch.Tensor, log_partial_hazard: torch.Tensor) -> torch.Tensor:
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
    if isinstance(log_partial_hazard, np.ndarray):
        log_partial_hazard = torch.from_numpy(log_partial_hazard)
    if isinstance(log_partial_hazard, pd.Series):
        log_partial_hazard = torch.tensor(log_partial_hazard.values)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    if isinstance(y, pd.Series):
        y = torch.tensor(y.values)

    time, event = transform_back_torch(y)
    partial_hazard = torch.exp(log_partial_hazard)
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
                likelihood = likelihood - torch.log(
                    risk_set_sum - ((ell / death_set_count) * death_set_risk)
                )
            risk_set_sum = risk_set_sum - accumulated_sum
            accumulated_sum = 0
            death_set_count = 0
            death_set_risk = 0

        if sample_event:
            death_set_count = death_set_count + 1
            death_set_risk = death_set_risk + sample_partial_hazard
            likelihood = likelihood + sample_partial_log_hazard

        accumulated_sum = accumulated_sum + sample_partial_hazard
        previous_time = sample_time

    for ell in range(death_set_count):
        likelihood = likelihood - torch.log(
            risk_set_sum - ((ell / death_set_count) * death_set_risk)
        )
    return -likelihood

class EfronLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        # Initialize any additional variables you need for your custom loss function here.

    def forward(self, prediction, input):
        loss = efron_likelihood_torch(input, prediction)
        return loss.to(torch.float32)

# Deephit Loss - use adapted pycox loss


def deephit_likelihood_1_torch(y, phi, duration_bins):
    #time, events = transform_back_torch_deephit(y)
    time, events = transform_back_torch(y)
    #time = time.reshape(time.shape[0],1)
    #print('shape idx dur befoer', idx_durations.shape)
    bins = duration_bins #torch.unique(time)
    idx_durations = (torch.bucketize(time, bins))
    idx_durations = idx_durations.view(-1, 1)
    #print('shape idx dur after', idx_durations.shape)
    # epsilon 
    epsilon = np.finfo(float).eps
    # pad phi as in pycox
    pad = torch.zeros_like(phi[:,:1])
    phi = torch.cat([phi, pad],axis=1)
    #print('phi shape', phi.shape)
    # create durations index
    bins = torch.unique(time)
    #print('idx_durations.max()',idx_durations.max())
    #print('time', time)
    #print('bins', bins)
    if phi.shape[1] <= idx_durations.max():
        raise ValueError(f"Network output `phi` is too small for `idx_durations`."+
                         f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`,"+
                         f" but got `phi.shape[1] = {phi.shape[1]}`")
    if events.dtype is torch.bool:
        events = events.float()
    #events = events.view(-1)
    #idx_durations = idx_durations.view(-1, 1)
    #phi = utils.pad_col(phi)

    gamma = phi.max(1)[0]
    #print('shapes', idx_durations.shape, phi.shape, gamma.shape, events.shape)
    cumsum = phi.sub(gamma.view(-1, 1)).exp().cumsum(1)
    sum_ = cumsum[:, -1]
    part1 = phi.gather(1, idx_durations).view(-1).sub(gamma).mul(events)
    part2 = - sum_.relu().add(epsilon).log()
    part3 = sum_.sub(cumsum.gather(1, idx_durations).view(-1)).relu().add(epsilon).log().mul(1. - events)
    # need relu() in part3 (and possibly part2) because cumsum on gpu has some bugs and we risk getting negative numbers.
    loss = - part1.add(part2).add(part3)
    return loss.sum()


class DeephitLoss(_Loss):
    def __init__(self, duration_bins = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        # Initialize any additional variables you need for your custom loss function here.
        self.duration_bins = duration_bins

    def forward(self, prediction, input):
        #print('forward prediction', prediction)
        #print('forward input', input)
        loss = deephit_likelihood_1_torch(input, prediction, self.duration_bins)
        return loss.to(torch.float32)

# def deephit_likelihood_1_torch(y, phi):
#     #time, events = transform_back_torch_deephit(y)
#     time, events = transform_back_torch(y)
#     #time = time.reshape(time.shape[0],1)
#     #print('shape idx dur befoer', idx_durations.shape)
#     bins = torch.unique(time)
#     idx_durations = (torch.bucketize(time, bins))
#     idx_durations = idx_durations.view(-1, 1)
#     #print('shape idx dur after', idx_durations.shape)
#     # epsilon 
#     epsilon = np.finfo(float).eps
#     # pad phi as in pycox
#     pad = torch.zeros_like(phi[:,:1])
#     phi = torch.cat([phi, pad],axis=1)
#     #print('phi shape', phi.shape)
#     # create durations index
#     bins = torch.unique(time)
#     #print('idx_durations.max()',idx_durations.max())
#     #print('time', time)
#     #print('bins', bins)
#     if phi.shape[1] <= idx_durations.max():
#         raise ValueError(f"Network output `phi` is too small for `idx_durations`."+
#                          f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`,"+
#                          f" but got `phi.shape[1] = {phi.shape[1]}`")
#     if events.dtype is torch.bool:
#         events = events.float()
#     #events = events.view(-1)
#     #idx_durations = idx_durations.view(-1, 1)
#     #phi = utils.pad_col(phi)

#     gamma = phi.max(1)[0]
#     #print('shapes', idx_durations.shape, phi.shape, gamma.shape, events.shape)
#     cumsum = phi.sub(gamma.view(-1, 1)).exp().cumsum(1)
#     sum_ = cumsum[:, -1]
#     part1 = phi.gather(1, idx_durations).view(-1).sub(gamma).mul(events)
#     part2 = - sum_.relu().add(epsilon).log()
#     part3 = sum_.sub(cumsum.gather(1, idx_durations).view(-1)).relu().add(epsilon).log().mul(1. - events)
#     # need relu() in part3 (and possibly part2) because cumsum on gpu has some bugs and we risk getting negative numbers.
#     loss = - part1.add(part2).add(part3)
#     return loss.sum()


# class DeephitLoss(_Loss):
#     def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
#         super().__init__(size_average, reduce, reduction)
#         # Initialize any additional variables you need for your custom loss function here.

#     def forward(self, prediction, input):
#         #print('forward prediction', prediction)
#         #print('forward input', input)
#         loss = deephit_likelihood_1_torch(input, prediction)
#         return loss.to(torch.float32)





# EH loss

PDF_PREFACTOR: float = 0.3989424488876037
SQRT_TWO: float = 1.4142135623730951
SQRT_EPS: float = 1.4901161193847656e-08
CDF_ZERO: float = 0.5

def bandwidth_function(time, event, n):
    return (8 * (sqrt(2) / 3)) ** (1 / 5) * n ** (-1 / 5)



def gaussian_integrated_kernel(x):
    return 0.5 * (1 + erf(x / SQRT_TWO))



def gaussian_kernel(x):
    return PDF_PREFACTOR * exp(-0.5 * (x**2))



def kernel(a, b, bandwidth):
    kernel_matrix: torch.tensor = torch.empty((a.shape[0], b.shape[0]))
    for ix in range(a.shape[0]):
        for qx in range(b.shape[0]):
            kernel_matrix[ix, qx] = gaussian_kernel(
                (a[ix] - b[qx]) / bandwidth
            )
    return kernel_matrix


def integrated_kernel(a, b, bandwidth):
    integrated_kernel_matrix: np.array = torch.empty(
        (a.shape[0], b.shape[0])
    )
    for ix in range(a.shape[0]):
        for qx in range(b.shape[0]):
            integrated_kernel_matrix[ix, qx] = gaussian_integrated_kernel(
                (a[ix] - b[qx]) / bandwidth
            )
    return integrated_kernel_matrix

def difference_kernels(a, b, bandwidth):
    difference: torch.tensor = torch.empty((a.shape[0], b.shape[0]))
    kernel_matrix: torch.tensor = torch.empty((a.shape[0], b.shape[0]))
    integrated_kernel_matrix: torch.tensor =torch.empty(
        (a.shape[0], b.shape[0])
    )
    for ix in range(a.shape[0]):
        for qx in range(b.shape[0]):
            difference[ix, qx] = (a[ix] - b[qx]) / bandwidth
            kernel_matrix[ix, qx] = gaussian_kernel(difference[ix, qx])
            integrated_kernel_matrix[ix, qx] = gaussian_integrated_kernel(
                difference[ix, qx]
            )

    return difference, kernel_matrix, integrated_kernel_matrix

# don't use this - wrong correct
def eh_likelihood_torch(
    # y and linear predictor contain two cols
    linear_predictor: torch.tensor,
    y: torch.tensor,
    sample_weight: torch.tensor = 1.0,
) -> torch.tensor:
    y1 = y[:,0]
    time, event = transform_back_torch(y1)
    # need two predictors here
    linear_predictor_1: torch.tensor = linear_predictor[:, 0] * sample_weight
    linear_predictor_2: torch.tensor = linear_predictor[:, 1] * sample_weight
    exp_linear_predictor_1 = torch.exp(linear_predictor_1)
    exp_linear_predictor_2 = torch.exp(linear_predictor_2)
    n_samples: int = time.shape[0]
    bandwidth = 1.30 * math.pow(n_samples, -0.2)
    #print(time.shape,exp_linear_predictor_1.shape )
    R_linear_predictor: torch.tensor = torch.log(time * exp_linear_predictor_1)
    inverse_sample_size_bandwidth: float = 1 / (n_samples * bandwidth)
    event_mask: torch.tensor = event.bool()

    _: torch.tensor
    kernel_matrix: torch.tensor
    integrated_kernel_matrix: torch.tensor

    (_, kernel_matrix, integrated_kernel_matrix,) = difference_kernels(
        a=R_linear_predictor,
        b=R_linear_predictor[event_mask],
        bandwidth=bandwidth,
    )
    #print('integrated_kernel_matrix', integrated_kernel_matrix)
    #print('(_, kernel_matrix, integrated_kernel_matrix,) ', (_, kernel_matrix, integrated_kernel_matrix,) )
    kernel_matrix = kernel_matrix[event_mask, :]

    inverse_sample_size: float = 1 / n_samples

    kernel_sum: torch.tensor = kernel_matrix.sum(axis=0)

    #print('(exp_linear_predictor_2 / exp_linear_predictor_1).repeat(torch.sum(event))',(exp_linear_predictor_2 / exp_linear_predictor_1)
    #  .repeat(torch.sum(event)))
    #print('torch.sum(event))',torch.sum(event))
    #!ATTENTION TORCH REPEAT DIFFERENT FROM NUMPY REPEAT
    # USE https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html
    integrated_kernel_sum: torch.tensor = (
        integrated_kernel_matrix
        * (exp_linear_predictor_2 / exp_linear_predictor_1)
        .repeat_interleave(torch.sum(event))
        .reshape(-1, torch.sum(event))
    ).sum(axis=0)
    #print('integrated_kernel_sum', integrated_kernel_sum)
    likelihood: torch.tensor = inverse_sample_size * (
        #linear_predictor_2[event_mask].sum()
        #- R_linear_predictor[event_mask].sum()
        #+ torch.log(inverse_sample_size_bandwidth * kernel_sum).sum()
        - torch.log(inverse_sample_size * integrated_kernel_sum[1:]).sum()
    )
    return -likelihood

# same using rv in pytorch
def eh_likelihood_torch_2(
    # y and linear predictor contain two cols
    linear_predictor: torch.tensor,
    y: torch.tensor,
    sample_weight: torch.tensor = 1.0,
    bandwidth: torch.tensor = None
) -> torch.tensor:
    #print('linear_predictor shape',linear_predictor.shape)
    #print('linear_predictor type',type(linear_predictor))
    #y1 = y[:,0]
    time, event = transform_back_torch(y)
    #time, event = transform_back_torch(y)
    # need two predictors here
    linear_predictor_1: torch.tensor = linear_predictor[:, 0] * sample_weight
    linear_predictor_2: torch.tensor = linear_predictor[:, 1] * sample_weight
    exp_linear_predictor_1 = torch.exp(linear_predictor_1)
    exp_linear_predictor_2 = torch.exp(linear_predictor_2)

    n_events: int = torch.sum(event)
    n_samples: int = time.shape[0]
    if not bandwidth:
        bandwidth = 1.30 * torch.pow(n_samples, torch.tensor(-0.2))
    #print('bandwidth', bandwidth)
    R_linear_predictor: torch.tensor = torch.log(time * exp_linear_predictor_1)
    inverse_sample_size_bandwidth: float = 1 / (n_samples * bandwidth)
    event_mask: torch.tensor = event.bool()

    _: torch.tensor
    kernel_matrix: torch.tensor
    integrated_kernel_matrix: torch.tensor

    inverse_sample_size_bandwidth: float = 1 / (n_samples * bandwidth)
    event_mask = event.bool()
    rv = torch.distributions.normal.Normal(0, 1, validate_args=None)
    sample_repeated_linear_predictor = (
        (exp_linear_predictor_2 / exp_linear_predictor_1).repeat((int(n_events.item()), 1)).T
    )
    diff = (
        R_linear_predictor.reshape(-1, 1) - R_linear_predictor[event_mask]
    ) / bandwidth

    kernel_matrix = torch.exp(
        -1 / 2 * torch.square(diff[event_mask, :])
    ) / torch.sqrt(torch.tensor(2) * torch.pi)
    integrated_kernel_matrix = rv.cdf(diff)
    
    inverse_sample_size: float = 1 / n_samples
    kernel_sum = kernel_matrix.sum(axis=0)
    integrated_kernel_sum = (
        sample_repeated_linear_predictor * integrated_kernel_matrix
    ).sum(axis=0)
    #print('integrated_kernel_sum', integrated_kernel_sum)
    # print(linear_predictor_2[event_mask].sum()/n_samples
    #     , R_linear_predictor[event_mask].sum()/n_samples
    #     , torch.log(inverse_sample_size_bandwidth * kernel_sum).sum()/n_samples
    #     , torch.log(inverse_sample_size * integrated_kernel_sum).sum()/n_samples)
    
    likelihood: torch.tensor = inverse_sample_size * (
        linear_predictor_2[event_mask].sum()
        - R_linear_predictor[event_mask].sum()
        + torch.log(inverse_sample_size_bandwidth * kernel_sum).sum()
        - torch.log(inverse_sample_size * integrated_kernel_sum).sum()
    )
    return -likelihood


class EHLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', bandwidth = None) -> None:
        super().__init__(size_average, reduce, reduction)
        # Initialize any additional variables you need for your custom loss function here.
        self.bandwidth = bandwidth

    def forward(self, prediction, input):
        loss = eh_likelihood_torch_2(prediction,input, bandwidth=self.bandwidth)
        #print('loss', loss)
        return loss


# AFT loss

def aft_likelihood_torch(linear_predictor: torch.Tensor,
    y: torch.Tensor, 
    sample_weight: torch.Tensor = 1.0, 
    bandwidth: torch.tensor = None) -> torch.Tensor:

    #print('y shape', y.shape)
    #print('linear predictor', linear_predictor.shape)

    if isinstance(linear_predictor, np.ndarray):
        linear_predictor = torch.from_numpy(linear_predictor)
    if isinstance(linear_predictor, pd.Series):
        linear_predictor = torch.tensor(linear_predictor.values)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    if isinstance(y, pd.Series):
        y = torch.tensor(y.values)

    if linear_predictor.ndim > 1:
        linear_predictor = linear_predictor.reshape(-1)

    time, event = transform_back_torch(y)
    #print('timeshape', time.shape)
    #print('eventshape', event.shape)
    n_samples: int = time.shape[0]
    if not bandwidth:
        bandwidth = 1.30 * math.pow(n_samples, -0.2)
    #print('bandwidth', bandwidth)  
    linear_predictor: torch.tensor = linear_predictor * sample_weight
    R_linear_predictor: torch.tensor = torch.log(
        time * torch.exp(linear_predictor)
    )
    inverse_sample_size_bandwidth: float = 1 / (n_samples * bandwidth)
    event_mask = event.bool()
    rv = torch.distributions.normal.Normal(0, 1, validate_args=None)
    #print('R_linear_predictor.reshape(-1, 1) shape',R_linear_predictor.reshape(-1, 1).shape)
    #print('R_linear_predictor.',R_linear_predictor.shape)
    #print('event_mask shape', event_mask.shape)
    # previously with reshape like in comment below
    # diff = (
    #     R_linear_predictor.reshape(-1, 1) - R_linear_predictor[event_mask]
    # ) / bandwidth
    diff = (
        R_linear_predictor.reshape(-1, 1) - R_linear_predictor[event_mask]
    ) / bandwidth

    kernel_matrix = torch.exp(
        -1 / 2 * torch.square(diff[event_mask, :])
    ) / torch.sqrt(torch.tensor(2) * torch.pi)

    integrated_kernel_matrix = rv.cdf(diff)

    inverse_sample_size: float = 1 / n_samples
    kernel_sum = kernel_matrix.sum(axis=0)
    integrated_kernel_sum = integrated_kernel_matrix.sum(axis=0)
    #print('integrated_kernel_matrix',integrated_kernel_matrix)

    likelihood = inverse_sample_size * (
        linear_predictor[event_mask].sum()
        - R_linear_predictor[event_mask].sum()
        + torch.log(inverse_sample_size_bandwidth * kernel_sum).sum()
        - torch.log(inverse_sample_size * integrated_kernel_sum).sum()
    )
    return -likelihood

class AFTLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', bandwidth = None) -> None:
        super().__init__(size_average, reduce, reduction)
        # Initialize any additional variables you need for your custom loss function here.
        self.bandwidth = bandwidth

    def forward(self, prediction, input): #add bandwidth and sample weight
        #print('input', input)
        #print('prediction', prediction)
        #if prediction.dtype == torch.float32:
        #    prediction = prediction.double()
        #if input.dtype == torch.float32:
        #    input = input.double()
        loss = aft_likelihood_torch(prediction, input, bandwidth=self.bandwidth)
        return loss


def ah_likelihood_torch(
    linear_predictor,
    y_torch,
    sample_weight=1.0,
    bandwidth = None

) -> torch.Tensor:
    
    if isinstance(linear_predictor, np.ndarray):
        linear_predictor = torch.from_numpy(linear_predictor)
    if isinstance(linear_predictor, pd.Series):
        linear_predictor = torch.tensor(linear_predictor.values)
    if isinstance(y_torch, np.ndarray):
        y_torch = torch.from_numpy(y_torch)
    if isinstance(y_torch, pd.Series):
        y_torch = torch.tensor(y_torch.values)

    if linear_predictor.ndim > 1:
        linear_predictor = linear_predictor.reshape(-1)

    #print('linear_predictor', linear_predictor)
    #print('y_torch', y_torch)

    time, event =  transform_back_torch(y_torch)
    n_samples: int = time.shape[0]
    n_events: int = torch.sum(event)
    linear_predictor: torch.tensor = linear_predictor * sample_weight
    if not bandwidth:
        bandwidth = 1.30 * torch.pow(n_samples, torch.tensor(-0.2))
    # R_linear_predictor: torch.tensor = torch.log(
    #     time * torch.exp(linear_predictor)
    # )
    R_linear_predictor: torch.tensor = torch.log(
        time) + linear_predictor
    
    inverse_sample_size_bandwidth: float = 1 / (n_samples * bandwidth)
    event_mask = event.bool()
    rv = torch.distributions.normal.Normal(0, 1, validate_args=None)
    sample_repeated_linear_predictor = (
        torch.exp(-linear_predictor).repeat((int(n_events.item()), 1)).T
    )

    diff = (
        R_linear_predictor.reshape(-1, 1) - R_linear_predictor[event_mask]
    ) / bandwidth

    kernel_matrix = torch.exp(
        -1 / 2 * torch.square(diff[event_mask, :])
    ) / torch.sqrt(torch.tensor(2) * torch.pi)
    integrated_kernel_matrix = rv.cdf(diff)
    #print('integrated kernel matrix', integrated_kernel_matrix)
    inverse_sample_size: float = 1 / n_samples
    kernel_sum = kernel_matrix.sum(axis=0)
    integrated_kernel_sum = (
        sample_repeated_linear_predictor * integrated_kernel_matrix
    ).sum(axis=0)
    likelihood = inverse_sample_size * (
        -R_linear_predictor[event_mask].sum()
        + torch.log(inverse_sample_size_bandwidth * kernel_sum).sum()
        - torch.log(inverse_sample_size * integrated_kernel_sum).sum()
    )
    return -likelihood

class AHLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', bandwidth = None) -> None:
        super().__init__(size_average, reduce, reduction)
        # Initialize any additional variables you need for your custom loss function here.
        self.bandwidth = bandwidth

    def forward(self, prediction, input): #add bandwidth and sample weight
        #print('input forward', input)
        #print('prediction forward', prediction)
        loss = ah_likelihood_torch(prediction, input, bandwidth=self.bandwidth)
        return loss
    

# class AFTLoss(_Loss):
#     def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', bandwidth = None) -> None:
#         super().__init__(size_average, reduce, reduction)
#         # Initialize any additional variables you need for your custom loss function here.
#         self.bandwidth = bandwidth

#     def forward(self, prediction, input): #add bandwidth and sample weight
#         #print('input', input)
#         #print('prediction', prediction)
#         #if prediction.dtype == torch.float32:
#         #    prediction = prediction.double()
#         #if input.dtype == torch.float32:
#         #    input = input.double()
#         loss = aft_likelihood_torch(prediction, input, bandwidth=self.bandwidth)
#         return loss

# Cindex

# TODO: Write in Pytorch, especially equivalent for add.reduceat()

# def KaplanMeier(time: np.array, event: np.array, 
#                 cens_dist: bool = False
# ) -> tuple[np.array, np.array] | tuple[np.array,np.array,np.array]:
#     """_summary_

#     Parameters
#     ----------
#     time : npt.NDArray[float]
#         _description_
#     event : npt.NDArray[int]
#         _description_
#     cens_dist : bool, optional
#         _description_, by default False

#     Returns
#     -------
#     tuple[npt.NDArray[float], npt.NDArray[float]] | tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[int]]
#         _description_
    
#     References
#     ----------
#     .. [1] Kaplan, E. L. and Meier, P., "Nonparametric estimation from incomplete observations",
#            Journal of The American Statistical Association, vol. 53, pp. 457-481, 1958.
#     .. [2] S. Pölsterl, “scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn,”
#            Journal of Machine Learning Research, vol. 21, no. 212, pp. 1–6, 2020.
#     """
#     # similar approach to sksurv, but no loops
#     # even and censored is other way round in sksurv ->clarify
#     #time, event = transform_back(y)
#     # order, remove later

#     # is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    
#     # if is_sorted(time) == False:
#     #     order = np.argsort(time, kind="mergesort")
#     #     time = time[order]
#     #     event = event[order]
    
#     times = torch.unique(time)
#     idx = torch.bucketize(time, np.unique(time))
#     # numpy diff nth discrete difference over index, add 1 at the beginning
#     breaks = torch.nonzero(torch.cat(([1], np.diff(idx)))).squeeze()

#     # flatnonzero return indices that are nonzero in flattened version
#     n_events = np.add.reduceat(event, breaks, axis=0)
#     n_at_risk = np.sum(np.unique((np.outer(time,time)>=np.square(time)).astype(int).T,axis=0),axis=1)[::-1]
    
#     # censoring distribution for ipcw estimation
#     #n_censored a vector, with 1 at censoring position, zero elsewhere
#     if cens_dist:
#         n_at_risk -= n_events
#         # for each unique time step how many observations are censored
#         censored = 1-event

#         n_censored = np.add.reduceat(censored, breaks, axis=0)
#         vals = 1-np.divide(
#         n_censored, n_at_risk,
#         out=np.zeros(times.shape[0], dtype=float),
#         where=n_censored != 0,
#     )
        
#         estimates = np.cumprod(vals)
#         return times, estimates, n_censored


#     else:
#         vals = 1-np.divide(
#         n_events, n_at_risk,
#         out=np.zeros(times.shape[0], dtype=float),
#         where=n_events != 0,
#         )
#         estimates = np.cumprod(vals)
#         return times, estimates

# def ipcw_estimate(time: np.array, event: np.array) -> tuple[np.array, np.array]:
#     """IPCW

#     Parameters
#     ----------
#     time : npt.NDArray[float]
#         _description_
#     event : npt.NDArray[int]
#         _description_

#     Returns
#     -------
#     tuple[npt.NDArray[float], npt.NDArray[float]]
#         _description_
#     """
#     time, event = time.to_numpy(), event.to_numpy()
#     unique_time, cens_dist, n_censored = KaplanMeier(time, event, cens_dist=True) 
#     #print(cens_dist)
#     # similar approach to sksurv
#     idx = np.searchsorted(unique_time, time)
#     est = 1.0/cens_dist[idx] # improve as divide by zero
#     est[n_censored[idx]!=0] = 0
#     # in R mboost there is a maxweight of 5
#     est[est>5] = 5
#     return torch.from_numpy(unique_time), torch.from_numpy(est)

# def compute_weights(y: npt.NDArray[float], approach: str='paper') -> npt.NDArray[float]:
#     """_summary_

#     Parameters
#     ----------
#     y : npt.NDArray[float]
#         Sorted array containing survival time and event where negative value is taken as censored event.
#     approach : str, optional
#         Choose mboost implementation or paper implementation of c-boosting, by default 'paper'.

#     Returns
#     -------
#     npt.NDArray[float]
#         Array of weights.

#     References
#     ----------
#     .. [1] 1. Mayr, A. & Schmid, M. Boosting the concordance index for survival data–a unified framework to derive and evaluate biomarker combinations. 
#        PloS one 9, e84483 (2014).

#     """
#     time, event = transform_back(y) 
#     n = event.shape[0]

#     _, ipcw_new = ipcw_estimate(time, event)

#     ipcw = ipcw_new #ipcw_old
#     survtime = time
#     wweights = np.full((n,n), np.square(ipcw)).T # good here


#     weightsj = np.full((n,n), survtime).T

#     weightsk = np.full((n,n), survtime) #byrow = TRUE in R, in np automatic no T required

#     if approach == 'mboost':
#         # implementing   weightsI <- ifelse(weightsj == weightsk, .5, (weightsj < weightsk) + 0) - diag(.5, n,n)
#         # from mboost github repo
#         weightsI = np.empty((n,n))
#         weightsI[weightsj == weightsk] = 0.5
#         weightsI = (weightsj < weightsk).astype(int)
#         weightsI = weightsI - np.diag(0.5*np.ones(n))
#     if approach == 'paper':
#         weightsI = (weightsj < weightsk).astype(int) 

#     wweights = wweights * weightsI 
    
#     wweights = wweights / np.sum(wweights)

#     return wweights



# def cind_loss(y: npt.NDArray[float], predictor: npt.NDArray[float], sigma: npt.NDArray[float] = 0.1) -> npt.NDArray[float]:
#     """Generate negative loglikelihood (loss) according to C-boosting model by Mayr and Schmid. Assumes times have been sorted beforehand.

#     Parameters
#     ----------
#     y : npt.NDArray[float]
#         Sorted array containing survival time and event where negative value is taken as censored event.
#     predictor : npt.NDArray[float]
#         Estimated hazard.
#     sigma : npt.NDArray[float], optional
#         _description_, by default 0.1.

#     Returns
#     -------
#     npt.NDArray[float]
#         Negative loglikelihood (loss) according to C-boosting model by Mayr and Schmid.

#     References
#     ----------
#     .. [1] 1. Mayr, A. & Schmid, M. Boosting the concordance index for survival data–a unified framework to derive and evaluate biomarker combinations. 
#        PloS one 9, e84483 (2014).
#     """
#     # f corresponds to predictor in paper
#     time, _ = transform_back(y)
#     n = time.shape[0]
#     etaj = np.full((n,n), predictor)
#     etak = np.full((n,n), predictor).T
#     x = (etak - etaj) 
#     weights_out = compute_weights(y)
#     c_loss = 1/(1+np.exp(x/sigma))*weights_out
#     return -np.sum(c_loss)