#cind
import numpy as np
import sys
import numpy.typing as npt
from xgbsurv.models.utils import  transform_back, ipcw_estimate


def compute_weights(y: npt.NDArray[float], approach: str='paper') -> npt.NDArray[float]:
    """_summary_

    Parameters
    ----------
    y : npt.NDArray[float]
        Sorted array containing survival time and event where negative value is taken as censored event.
    approach : str, optional
        Choose mboost implementation or paper implementation of c-boosting, by default 'paper'.

    Returns
    -------
    npt.NDArray[float]
        Array of weights.

    References
    ----------
    .. [1] 1. Mayr, A. & Schmid, M. Boosting the concordance index for survival data–a unified framework to derive and evaluate biomarker combinations. 
       PloS one 9, e84483 (2014).

    """
    time, event = transform_back(y) 
    n = event.shape[0]

    _, ipcw_new = ipcw_estimate(time, event)

    ipcw = ipcw_new #ipcw_old
    survtime = time
    wweights = np.full((n,n), np.square(ipcw)).T # good here


    weightsj = np.full((n,n), survtime).T

    weightsk = np.full((n,n), survtime) #byrow = TRUE in R, in np automatic no T required

    if approach == 'mboost':
        # implementing   weightsI <- ifelse(weightsj == weightsk, .5, (weightsj < weightsk) + 0) - diag(.5, n,n)
        # from mboost github repo
        weightsI = np.empty((n,n))
        weightsI[weightsj == weightsk] = 0.5
        weightsI = (weightsj < weightsk).astype(int)
        weightsI = weightsI - np.diag(0.5*np.ones(n))
    if approach == 'paper':
        weightsI = (weightsj < weightsk).astype(int) 

    wweights = wweights * weightsI 
    
    wweights = wweights / np.sum(wweights)

    return wweights



def cind_loss(y: npt.NDArray[float], predictor: npt.NDArray[float], sigma: npt.NDArray[float] = 0.1) -> npt.NDArray[float]:
    """Generate negative loglikelihood (loss) according to C-boosting model by Mayr and Schmid. Assumes times have been sorted beforehand.

    Parameters
    ----------
    y : npt.NDArray[float]
        Sorted array containing survival time and event where negative value is taken as censored event.
    predictor : npt.NDArray[float]
        Estimated hazard.
    sigma : npt.NDArray[float], optional
        _description_, by default 0.1.

    Returns
    -------
    npt.NDArray[float]
        Negative loglikelihood (loss) according to C-boosting model by Mayr and Schmid.

    References
    ----------
    .. [1] 1. Mayr, A. & Schmid, M. Boosting the concordance index for survival data–a unified framework to derive and evaluate biomarker combinations. 
       PloS one 9, e84483 (2014).
    """
    # f corresponds to predictor in paper
    time, event = transform_back(y)
    n = time.shape[0]
    n_events = np.sum(event)
    etaj = np.full((n,n), predictor)
    etak = np.full((n,n), predictor).T
    x = (etak - etaj) 
    weights_out = compute_weights(y)
    c_loss = 1/(1+np.exp(x/sigma))*weights_out
    return np.sum(c_loss)*n_events #/y.shape[0] # compared with R funciton - looks good

def cind_gradient(y, predictor, weights, sigma=0.1):

    time, event = transform_back(y)
    n_events = np.sum(event)
    n = time.shape[0]
    etaj = np.full((n,n), predictor) # looks good
    etak = np.full((n,n), predictor).T # looks good
    x = (etak - etaj) 
    weights_out = weights #compute_weights(time, event)
    M1 = np.exp(x/sigma)/(sigma *np.square((1+np.exp(x/sigma))))*weights_out #verify squared, reckon elementwise
    cind_grad = np.sum(M1,axis=0) - np.sum(M1, axis=1)
    return cind_grad*n_events # beware of negative sign

def cind_hessian(y, predictor, weights, sigma=0.1):
    time, event = transform_back(y)
    n_events = np.sum(event)
    #print(time)
    n = time.shape[0]
    etaj = np.full((n,n), predictor)
    etak = np.full((n,n), predictor).T
    x = (etak - etaj)
    # factor for equation from B.14 in thesis
    factor = weights/sigma #compute_weights(time, event)
    # 1. first part of equation
    # formula taken literally, simplification possible
    H = factor*(np.exp(x/sigma)*(-1/sigma)*(1/np.square(1+np.exp(x/sigma)))+np.exp(x/sigma)*(-2)*(1/np.power((1+np.exp(x/sigma)),3))*np.exp(x/sigma)*(1/sigma))
    # would potentially need to sum the eq elements
    # 1. axis=0, 2. axis=1
    # add the minus
    c_hessian = np.sum(H, axis=0) + np.sum(H, axis=1)
    #c_hessian = (np.exp(x/sigma)*(-1/sigma)*(1/np.square(1+np.exp(x/sigma)))+np.exp(x/sigma)*(-2)*(1/np.power((1+np.exp(x/sigma)),3))*(1/np.square(1+np.exp(x/sigma)))*(-1/sigma))*weights_out #verify squared, reckon elementwise
    #r = np.zeros((n,n))
    #np.fill_diagonal(r,1)
    # in xgboost hessian is diagonal vector hessian matrix
    # r = np.ones(n)
    return -c_hessian*n_events #r #

def cind_objective(y: np.array, predictor: np.array) -> tuple[np.array, np.array]:
    """Objective function calculating gradient and hessian of C-boosting negative likelihood function. Assumes data is sorted.

    Parameters
    ----------
    y : npt.NDArray[float]
        Sorted array containing survival time and event where negative value is taken as censored event.
    log_partial_hazard : npt.NDArray[float]
        Estimated hazard.

    Returns
    -------
    gradient : npt.NDArray[float]
        Gradient of the C-boosting negative likelihood (loss) function.
    hessian : npt.NDArray[float]]
        Diagonal Hessian of the C-boosting negative (loss) likelihood.

    References
    ----------
    .. [1] 1. Mayr, A. & Schmid, M. Boosting the concordance index for survival data–a unified framework to derive and evaluate biomarker combinations. 
       PloS one 9, e84483 (2014).
    """

    weights = compute_weights(y)
    gradient = cind_gradient(y, predictor, weights = weights, sigma=0.1)
    hessian = cind_hessian(y, predictor, weights = weights, sigma=0.1)
    return gradient, hessian



class CindPredictor():
    """Prediction functions particular to the Cox PH model"""
    
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