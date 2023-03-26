from xgbsurv.models.breslow_final import breslow_likelihood, breslow_objective, \
BreslowPredictor
from xgbsurv.models.efron_final import efron_likelihood, efron_objective, \
EfronPredictor
from xgbsurv.models.cind_final import cind_loss, cind_objective, \
CindPredictor
from xgbsurv.models.deephit_pycox_final import deephit_loss1_pycox, deephit_pycox_objective, \
DeephitPredictor
from xgbsurv.models.eh_aft_final import aft_likelihood, aft_objective, \
AftPredictor
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# dicts of objective, loss and prediction functions

loss_dict= {'breslow_loss': breslow_likelihood, 'efron_loss': efron_likelihood, \
            'cind_loss': cind_loss, 'deephit_loss':deephit_loss1_pycox, 'aft_loss':aft_likelihood}
objective_dict= {'breslow_objective': breslow_objective, 'efron_objective': efron_objective, \
                 'cind_objective': cind_objective, 'deephit_objective':deephit_pycox_objective, 'aft_objective':aft_objective}
pred_dict = {'breslow_objective': BreslowPredictor, 'efron_objective': EfronPredictor,\
              'cind_objective': CindPredictor, 'deephit_objective': DeephitPredictor, 'aft_objective':AftPredictor}



class XGBSurv(XGBRegressor):

    # TODO: set standard objective to Efron.
    def __init__(self, *, objective=None, eval_metric=None, **kwargs) -> None:
        self.cum_hazard_baseline = None
        self.model_type = None
        if objective in objective_dict:
            obj = objective_dict[objective]
            self.model_type = objective
        elif callable(objective):
            obj = objective
        else:
            obj = objective

        if eval_metric in loss_dict:
            eval_loss = loss_dict[eval_metric]
        elif callable(objective):
            eval_loss = eval_metric
        else:
            eval_loss = eval_metric 
        #eval_loss = loss_dict['custom_loss']

        super().__init__(objective=obj, eval_metric= eval_loss, **kwargs)
        #disable_default_eval_metric=disable,

    def fit(self, X, y, *, eval_test_size=None, **kwargs):

        X, y = self._sort_X_y(X,y)
        self.y = y
        # deephit multioutput
        # so far this creates errors
        # if self.model_type=='deephit_objective':
        #     print('running deephit')
        #     n = len(np.unique(np.abs(y)))
        #     y = np.tile(y, (n,1)).T

        if eval_test_size is not None:
        
            params = super(XGBRegressor, self).get_xgb_params()
            
            #TODO: Potentially swap for k-fold to have better event distribution
            #TODO: verify for deephit split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=eval_test_size, random_state=params['random_state'])
            
            #eval_set = [(X_test, y_test)]
            
            # Could add (X_train, y_train) to eval_set 
            # to get .eval_results() for both train and test
            eval_set = [(X_train, y_train),(X_test, y_test)] 
            
            kwargs['eval_set'] = eval_set
            
            return super(XGBSurv, self).fit(X_train, y_train, **kwargs)
        else:
            return super(XGBSurv, self).fit(X, y, **kwargs)
        
    def predict_cumulative_hazard_function(self, X, dataframe=False):
        if self.model_type:
            # TODO: Set output margin
            self.pred_hazards = super(XGBSurv, self).predict(X, output_margin=True)
            self.predictor = pred_dict[str(self.model_type)]
            # fit only with training hazards?
            est = self.predictor().fit(self.pred_hazards, self.y) # y from training set, cache from fit(), call breslow in fit, efron as well
            self.uniq_times, self.cum_hazard_baseline = est.get_cumulative_hazard_function()
            # sample hazards have to repeated to get prediction for each individual
            self.pred_cum_hazards = np.outer(np.exp(self.pred_hazards), self.cum_hazard_baseline)
            times = np.tile(self.uniq_times,(self.pred_cum_hazards.shape[0],1))
            if dataframe:
                d = dict()
                d['time'] = self.uniq_times
                for ind, val in enumerate(times):
                    d['patient_'+str(ind)] = self.pred_cum_hazards[ind]
                df = pd.DataFrame(d)
                return df
            else:
                return times, self.pred_cum_hazards
        else:
            raise NotImplementedError("Cumulative hazard not applicable to the model you provided.")
    
    def predict_survival_function(self, X, dataframe=False):

        if self.cum_hazard_baseline is None:
            self.pred_hazards = super(XGBSurv, self).predict(X, output_margin=True)
            self.predictor = pred_dict[str(self.model_type)]
            # fit only with training hazards?
            est = self.predictor().fit(self.pred_hazards, self.y) # y from training set, cache from fit(), call breslow in fit, efron as well
            self.uniq_times, self.cum_hazard_baseline = est.get_cumulative_hazard_function()

        self.pred_survival = np.exp(-self.pred_cum_hazards)
        times = np.tile(self.uniq_times,(self.pred_survival.shape[0],1))
        if dataframe:
            d = dict()
            d['time'] = self.uniq_times
            for ind, val in enumerate(times):
                d['patient_'+str(ind)] = self.pred_survival[ind]
            df = pd.DataFrame(d)
            return df
        else:
            return self.uniq_times, self.pred_survival

        
    def get_loss_functions(self):
        return loss_dict
    
    def get_objective_functions(self):
        return objective_dict
    
    def _sort_X_y(self, X, y):
        # naming convention as in sklearn
        # maybe move to utils
        # add sorting here, maybe there is a faster way
        is_sorted = lambda a: np.all(a[:-1] <= a[1:])
        y_abs = abs(y)
        if is_sorted(y_abs) is False:
            order = np.argsort(y_abs, kind="mergesort")
            y = y[order]
            X = X[order]
        return X, y
        
