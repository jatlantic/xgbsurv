from xgbsurv.models.breslow_final import breslow_likelihood, breslow_objective, \
 get_cumulative_hazard_function_breslow
from xgbsurv.models.efron_final import efron_likelihood, efron_objective, \
 get_cumulative_hazard_function_efron
from xgbsurv.models.cind_final import cind_loss, cind_objective 
from xgbsurv.models.deephit_pycox_final import deephit_loss1_pycox, deephit_pycox_objective 
from xgbsurv.models.eh_aft_final import aft_likelihood, aft_objective, \
 get_cumulative_hazard_function_aft
from xgbsurv.models.eh_ah_final import ah_likelihood, ah_objective, \
 get_cumulative_hazard_function_ah
from xgbsurv.models.eh_final import eh_likelihood, eh_objective,\
      get_cumulative_hazard_function_eh
from xgbsurv.models.utils import transform_back, sort_X_y_pandas
from xgbsurv.docstrings.xgbsurv_docstrings import get_xgbsurv_docstring, \
get_xgbsurv_fit_docstring
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



# dicts of objective, loss and prediction functions
loss_dict = {
            'breslow_loss': breslow_likelihood, 
            'efron_loss': efron_likelihood, 
            'cind_loss': cind_loss, 
            'deephit_loss':deephit_loss1_pycox, 
            'aft_loss':aft_likelihood, 
            'ah_loss':ah_likelihood, 
            'eh_loss': eh_likelihood
            }


objective_dict = {
            'breslow_objective': breslow_objective,
            'efron_objective': efron_objective, 
            'cind_objective': cind_objective, 
            'deephit_objective':deephit_pycox_objective, 
            'aft_objective':aft_objective, 
            'ah_objective':ah_objective,
            'eh_objective': eh_objective
            }

pred_dict =  {
            'breslow_objective': get_cumulative_hazard_function_breslow, 
            'efron_objective': get_cumulative_hazard_function_efron, 
            'aft_objective': get_cumulative_hazard_function_aft,
            'ah_objective': get_cumulative_hazard_function_ah,
            'eh_objective': get_cumulative_hazard_function_eh,
            }

class XGBSurv(XGBRegressor):
    """XGBSurv - Gradient Boosted Decision Trees for Survival Analysis."""
    __doc__ = get_xgbsurv_docstring()

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

        super().__init__(objective=obj, eval_metric= eval_loss, **kwargs)
        #disable_default_eval_metric=disable,

    def fit(self, X, y, *, eval_test_size=None, **kwargs):
        __doc__ = get_xgbsurv_fit_docstring()

        #print('types',type(X),type(y))
        #Ct transforms to numpy array to mixtures are expected
        if isinstance(X, np.ndarray) and isinstance(y, pd.Series):
            y = y.values
        
        if isinstance(X, np.ndarray) and isinstance(y, pd.DataFrame):
            y = y.values

        if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
            X, y = sort_X_y_pandas(X, y)
        elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
            X, y = self._sort_X_y(X, y)
        else:
            print("Data type is not correct - use either pandas DataFrame/Series or numpy ndarray. ")

        if eval_test_size is not None:
        
            params = super(XGBRegressor, self).get_xgb_params()
            
            #TODO: verify for deephit split
            #target_sign = np.sign(y) beware of deephit dims
            X_train, X_test, y_train, y_test = train_test_split(
                                                X, 
                                                y, 
                                                test_size=eval_test_size,
                                                random_state=params['random_state'],
                                                stratify=np.sign(y)) 
            #print('types2',type(X_train),type(y_train))
            #print('types3',type(X_test),type(y_test))
            #print('shapes', X_train.shape,y_train.shape)
            #print('shapes 2', X_test.shape,y_test.shape)
            if isinstance(X_train, pd.DataFrame) and isinstance(y_train, pd.Series):
                X_train, y_train = sort_X_y_pandas(X_train, y_train)

            elif isinstance(X_train, np.ndarray) and isinstance(y_train, np.ndarray):
                X_train, y_train = self._sort_X_y(X_train, y_train)

            elif isinstance(X_test, pd.DataFrame) and isinstance(y_test, pd.Series):
                X_test, y_test = sort_X_y_pandas(X_test, y_test)

            elif isinstance(X_test, np.ndarray) and isinstance(y_test, np.ndarray):
                X_test, y_test = self._sort_X_y(X_test, y_test)
            # eh case
            elif isinstance(X_train, pd.DataFrame) and isinstance(y_train, pd.DataFrame):
                X_train, y_train = sort_X_y_pandas(X_train, y_train)
            
            elif isinstance(X_test, pd.DataFrame) and isinstance(y_test, pd.DataFrame):
                X_test, y_test = sort_X_y_pandas(X_test, y_test)

            else:
                print("Data type is not correct - use either pandas DataFrame/Series or numpy ndarray. ")
            

            # 1. column training loss
            # 2. column separat validation set loss
            eval_set = [(X_train, y_train),(X_test, y_test)]
            #print('eval_set',eval_set)
            kwargs['eval_set'] = eval_set
            
            return super(XGBSurv, self).fit(X_train, y_train, **kwargs)
        else:
            X, y = self._sort_X_y(X,y)
            return super(XGBSurv, self).fit(X, y, **kwargs)
        
    # TODO: DataFrame Option
    def predict_cumulative_hazard_function(
        self, 
        X_train: np.array,
        X_test: np.array,
        y_train: np.array,
        y_test: np.array, 
        ) -> pd.DataFrame:
        """Obtain cumulative hazard function from your model."""
        if self.model_type:

            train_pred_hazards = super(XGBSurv, self).predict(X_train, output_margin=True)
            test_pred_hazards = super(XGBSurv, self).predict(X_test, output_margin=True)
            cum_hazard_predictions = pred_dict[str(self.model_type)](
                X_train, 
                X_test, 
                y_train, 
                y_test,
                train_pred_hazards, 
                test_pred_hazards
            )
            return cum_hazard_predictions
        else:
            raise NotImplementedError("Cumulative hazard not applicable to the model you provided.")
    
    # TODO: add model condition
    def predict_survival_function(
            self, 
            X_train: np.array, 
            X_test: np.array,
            y_train: np.array, 
            y_test: np.array):
        """Obtain survival function from your model."""
        
        X_train, y_train = self._sort_X_y(X_train, y_train)
        X_test, y_test = self._sort_X_y(X_test, y_test)
        df_cumulative_hazard = self.predict_cumulative_hazard_function(X_train, 
            X_test, y_train, y_test)
        return np.exp(-df_cumulative_hazard)
        
    def get_loss_functions(self):
        """Get implemented survival loss functions."""
        return loss_dict
    
    def get_objective_functions(self):
        """Get implemented survival objective functions."""
        return objective_dict
    
    def _sort_X_y(self, X, y):
        """Sort X, y data by absolute time."""
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
        if not isinstance(y, np.ndarray):
            print('y error',y, type(y))
            raise ValueError(f'y is not numpy.ndarray. Got {type(y)}.')
        #new condition begin
        if y.ndim > 1:
            #print("Array has more than one dimension.")
            # Check if the array has more than one column
            if y.shape[1] > 1:
                y_abs = y[:,0]
        else:        
            y_abs = y.copy()        
        y_abs = np.absolute(y_abs)
        #condition end
        #y_abs = np.absolute(y)
        if not np.all(np.diff(y_abs) >= 0):
            #print('Values are being sorted!')
            order = np.argsort(y_abs, kind="mergesort")
            y = y[order]
            X = X[order]
        return X, y
            
