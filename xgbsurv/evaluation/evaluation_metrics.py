from xgbsurv.evaluation.utils import transform_back
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.util import Surv
import numpy as np


def cindex_censored(target, estimate):
    event_time, event_indicator = transform_back(target)
    cindex = concordance_index_censored(event_indicator.astype('bool'), event_time, estimate)[0]
    return cindex
    

def ibs(target_train, target_test, surv_preds, times):
    # input numpy arrays
    # TODO: pandas input
    time_train, event_train = transform_back(target_train)
    time_test, event_test = transform_back(target_test)
    survival_train = Surv.from_arrays(event_train, time_train)
    survival_test = Surv.from_arrays(event_test, time_test)
    preds = np.asarray([[fn(t) for t in times] for fn in surv_preds])
    ibs_score = integrated_brier_score(survival_train, survival_test, preds, times)
    return ibs_score


