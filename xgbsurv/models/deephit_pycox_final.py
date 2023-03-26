import numpy as np
from xgbsurv.models.utils import transform, transform_back



def relu(x):
    return (np.maximum(0,x))

def deephit_loss1_pycox(y, phi)->float:
    """Pycox version of Deephit loss (1).

    Parameters
    ----------
    y : npt.NDArray[float]
        _description_
    log_partial_hazard : npt.NDArray[float]
        _description_

    Returns
    -------
    float
        _description_
    
    References
    ----------
    .. [1] Lee, C., Zame, W., Yoon, J. & Van Der Schaar, M. Deephit: A deep learning approach to survival analysis with competing risks. 
           in Proceedings of the AAAI conference on artificial intelligence vol. 32 (2018).
    .. [2] Liu, P., Fu, B. & Yang, S. X. HitBoost: survival analysis via a multi-output gradient boosting decision tree method. 
           IEEE Access 7, 56785–56795 (2019).
    """ 
    # follows Pycox Loss 1:1
    # log_partial_hazard -> matrix with row for each individual i - verify
    # or consider different order
    # x is the vector of covariates
    # K is the number of possible causes, here we consider only one cause
    # s is the time
    #time, event = transform_back(y)
    #event = event.dtype('float32')
    print('phi', phi)
    y = y.reshape(phi.shape)
    print('loss y shape', y.shape)
    y = y[:,0]
    time, event = transform_back(y)
    epsilon = 1e-7
    bins = np.unique(time)
    # let's assume idx_durations maps each time to the position in the unique time vector
    # no.digitize has no zero indexed group, therefore -1
    idx_durations_part1 = np.digitize(time, bins)-1
    idx_durations_part2 = np.digitize(time, bins)
    idx_durations_part1 = idx_durations_part1.reshape(-1, 1)
    idx_durations_part2 = idx_durations_part2.reshape(-1, 1)
    # padding as in pycox
    pad = np.zeros_like(phi[:,:1])
    phi = np.concatenate((phi, pad),axis=1)

    gamma = np.amax(phi, axis=1)

    intermed = np.exp(phi-gamma.reshape(-1, 1))
    cumsum  = intermed.cumsum(axis=1)
    #print('cumsum', cumsum)
    sum_ = cumsum[:, -1]
    #print('np.take_along_axis(phi, idx_durations, axis=1).reshape(-1).sub(gamma)',np.take_along_axis(phi, idx_durations, axis=1).reshape(-1)-gamma)
    #print('gamma', gamma)
    part1 = (np.take_along_axis(phi, idx_durations_part1, axis=1).reshape(-1)-gamma)*event
    #print('part1',part1)
    part2 = -np.log(np.maximum(0,sum_)+epsilon)
    #print('part2',part2)
    intermed = sum_-(np.take_along_axis(cumsum, idx_durations_part1, axis=1).reshape(-1))
    part3 = np.log(relu(intermed)+epsilon)*(1. -event)
    #print('part3',part3)
    loss = - (part1 + part2 + part3) # beware minus sign
    deephit_loss_1 = np.sum(loss)
    print('deephit_loss_1', deephit_loss_1)
    return deephit_loss_1



# calculate gradient
def pycox_gradient_no_gamma(y, phi):
    time, event = transform_back(y)
    # epsilon 
    epsilon = np.finfo(float).eps
    # pad phi as in pycox
    pad = np.zeros_like(phi[:,:1])
    phi = np.concatenate((phi, pad),axis=1)
    # create durations index
    bins = np.unique(time)
    idx_durations_part1 = (np.digitize(time, bins)-1).reshape(-1, 1)
    idx_durations_part2 = (np.digitize(time, bins)).reshape(-1, 1)
    n,k = phi.shape
    part1 = np.zeros_like(phi)
    np.put_along_axis(part1, idx_durations_part1,event.reshape(n, 1), axis=1)
    #part1 = event.reshape(n, 1)
    #print('part1', part1)
    phi_exp = np.exp(phi)
    cumsum = np.cumsum(phi_exp, axis=1)
    #print(cumsum[:,[-1]].shape)
    #print(np.take_along_axis(cumsum, idx_durations_part2, axis=1).shape)
    denominator_part2 = cumsum[:,[-1]]-np.take_along_axis(cumsum, idx_durations_part1, axis=1) #+epsilon
    #denominator_part2[denominator_part2==0.] = 1.0
    indicator_phi_q_larger_k1 = (np.arange(phi.shape[1]) > idx_durations_part1.reshape(-1, 1)).astype(int) # \leq not >, therefore idxpart1
    #print(indicator_phi_q_larger_k1)
    nominator_part2 = indicator_phi_q_larger_k1*phi_exp
    #print('nominator_part2',nominator_part2)
    #print('denominator_part2',denominator_part2)
    # correct the where function adapted for different idex_durations
    part2 = (1.0-event).reshape(n, 1)*np.divide(nominator_part2, denominator_part2, where=denominator_part2!=0. )
    #print('part2', part2)
    denominator_part3 = cumsum[:,[-1]]
    nominator_part3 = phi_exp
    part3 = nominator_part3/denominator_part3
    #print('part3', part3)
    result = -(part1+part2-part3)
    return result[:,:-1]


def pycox_second_deriv_no_gamma(y, phi):
    time, event = transform_back(y)
    # epsilon 
    epsilon = np.finfo(float).eps
    # pad phi as in pycox
    pad = np.zeros_like(phi[:,:1])
    phi = np.concatenate((phi, pad),axis=1)
    # create durations index
    bins = np.unique(time)
    idx_durations_part1 = (np.digitize(time, bins)-1).reshape(-1, 1)
    idx_durations_part2 = (np.digitize(time, bins)).reshape(-1, 1)
    n,k = phi.shape
    phi_exp = np.exp(phi)
    cumsum = np.cumsum(phi_exp, axis=1)
    denominator_part1 = cumsum[:,[-1]]-np.take_along_axis(cumsum, idx_durations_part1, axis=1) #+epsilon
    #print(denominator_part1)
    indicator_phi_q_larger_k1 = (np.arange(phi.shape[1]) > idx_durations_part1.reshape(-1, 1)).astype(int) # \leq not >, therefore idxpart1
    
    subpart1 = (1.0-event).reshape(n, 1)*(((indicator_phi_q_larger_k1*phi_exp)/denominator_part1)-((indicator_phi_q_larger_k1*np.square(phi_exp))/np.square(denominator_part1)))
    #print('subpart1',subpart1)
    denominator_part2 = cumsum[:,[-1]]
    subpart2 = (phi_exp/denominator_part2)-(np.square(phi_exp)/np.square(denominator_part2))
    #print('subpart2',subpart2)
    part1 = subpart1-subpart2
    return part1

def deephit_pycox_objective(y, phi):
    print(phi.shape)
    y = y.reshape(phi.shape)
    y = y[:,0]
    print(y.shape)
    grad = pycox_gradient_no_gamma(y, phi).reshape(-1)
    hess = pycox_second_deriv_no_gamma(y, phi).reshape(-1)
    hess = np.ones(grad.shape)
    print(grad.shape)
    print(hess.shape)
    return grad, hess


class DeephitPredictor():
    """Prediction functions particular to the Deephit model"""
    
    def __init__(self) -> None:
        self.uniq_times = None
        self.cum_hazard_baseline = None
        self.baseline_survival = None
        
    
    def fit(self, partial_hazard, y):
        raise NotImplementedError("This model does not provide for the function you asked for!")

    def get_cumulative_hazard_function(self):
        raise NotImplementedError("This model does not provide for the function you asked for!")

    def get_survival_function(self):
        raise NotImplementedError("This model does not provide for the function you asked for!")

    def get_survival_function_own(self, partial_hazard):
        raise NotImplementedError("This model does not provide for the function you asked for!")