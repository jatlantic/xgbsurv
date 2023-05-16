#from numba import jit
import numpy as np
import pandas as pd
# using numpy.typing.NDArray[A] as an alias for numpy.ndarray[Any, numpy.dtype[A]]:
# discuss adding dimensionality to it
# https://stackoverflow.com/questions/71109838/numpy-typing-with-specific-shape-and-datatype
import numpy.typing as npt


#@jit(nopython=True)
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



#@jit(nopython=True) # not really needed
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

