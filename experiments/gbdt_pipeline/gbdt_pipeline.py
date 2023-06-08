# gbdt pipeline and model
import pickle
import os
from xgbsurv.datasets import (load_metabric, load_flchain, load_rgbsg, load_support, load_tcga)
from xgbsurv import XGBSurv
import numpy as np
import pandas as pd
from scipy.stats import uniform as scuniform
from scipy.stats import randint as scrandint
from scipy.stats import loguniform as scloguniform 
from sklearn.metrics import make_scorer
from xgbsurv.evaluation import cindex_censored
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.decomposition import PCA
# import models
from xgbsurv.models.eh_aft_final import aft_likelihood, get_cumulative_hazard_function_aft
from pycox.evaluation import EvalSurv
from xgbsurv.models.utils import sort_X_y_pandas, transform_back, transform
from xgbsurv.preprocessing.dataset_preprocessing import discretizer_df
from sklearn.utils.fixes import loguniform
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

# path
current_path = os.getcwd()  # Get the current working directory path
#pickle_path = current_path+'/Documents/xgbsurv/experiments/params/'
print('current path', current_path)
somepath = os.path.abspath(os.path.join(current_path,  ".."))
pickle_path = somepath+'/params/'

loss_dict = {
            'breslow': breslow_likelihood, 
            'efron': efron_likelihood, 
            'cind': cind_loss, 
            'deephit':deephit_loss1_pycox, 
            'aft':aft_likelihood, 
            'ah':ah_likelihood, 
            'eh': eh_likelihood
            }

cum_hazard_dict ={
            'breslow': get_cumulative_hazard_function_breslow, 
            'efron': get_cumulative_hazard_function_efron, 
            'cind': cind_loss, 
            #'deephit':deephit_loss1_pycox, 
            'aft': get_cumulative_hazard_function_aft, 
            'ah':get_cumulative_hazard_function_ah, 
            'eh': get_cumulative_hazard_function_eh
                }

# custom splitter

class CustomSplit(StratifiedKFold):
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y, groups=None):
        print('split type',type(y))
        if isinstance(y, pd.DataFrame):
            y = y.values[:,0]
        elif isinstance(y, pd.Series):
            try:
                y = y.values[:,0]
            except:
                y = y.values
        elif isinstance(y, np.ndarray):
            try:
                if y.shape[1]>1:
                    y = y[:,0]
            except:
                pass

        bins = np.sign(y)
        return super().split(X, bins, groups=groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits





def train_gbdt_complete(
            dataset_name,
            X, 
            y, 
            i, 
            train_index, 
            test_index, 
            model='breslow',
            n_iter=50,
            tcga=False, 
            rand_state=42, 
            early_stopping_rounds=10, 
            base_score=0.0, 
            validation_size=0.2,
            verbose=1, 
            one_hot=False, 
            n_inner_splits=5,

               ):
    # model options: breslow, efron, eh, aft, ah, cboost
    # read params from pickle
    np.random.seed(rand_state)
    method = '_gbdt_'
    if tcga==True:
        with open(pickle_path+model+'_gbdt_tcga.pkl', 'rb') as file:
            params = pickle.load(file)
    else:
        with open(pickle_path+model+'_gbdt_not_tcga.pkl', 'rb') as file:
            params = pickle.load(file)
    # column transformer
    if one_hot:
        ct = make_column_transformer(
            (StandardScaler(), make_column_selector(dtype_include=['float32'])),
            (OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist'), make_column_selector(dtype_include=['category', 'object'])),
            remainder='passthrough')
    else:
        ct = make_column_transformer(
            (StandardScaler(), make_column_selector(dtype_include=['float32'])),
            remainder='passthrough')
        
    # define scoring function
    def custom_scoring_function(y_true, y_pred, model):

        if not isinstance(y_true, np.ndarray):
            y_true = y_true.values
        if not isinstance(y_pred, np.ndarray):
            y_pred = y_pred.values

        score = loss_dict[model](y_true, y_pred)

        return score

    scoring_function = make_scorer(custom_scoring_function,model=model, greater_is_better=False)

    # custom splitter
    inner_custom_cv = CustomSplit(n_splits=n_inner_splits, shuffle=True, random_state=rand_state)

    # set XGBSurv model
    estimator = XGBSurv(
        objective=model+'_objective',
        eval_metric=model+'_loss',
        random_state=rand_state, 
        disable_default_eval_metric=True,
        early_stopping_rounds=early_stopping_rounds, 
        base_score=base_score,
                    )
    #print('estimator', estimator)
    pl = Pipeline([('scaler',ct),
                    ('estimator', estimator)])
        
    rs = RandomizedSearchCV(pl, params, scoring = scoring_function, n_jobs=-1, 
                                cv=inner_custom_cv, n_iter=n_iter, refit=True, 
                                random_state=rand_state, verbose=verbose,
                                error_score = 'raise')
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    X_train, y_train = sort_X_y_pandas(X_train, y_train)
    X_test, y_test = sort_X_y_pandas(X_test, y_test)
    #print(X_train,y_train)

    # save splits and data 
    np.savetxt(current_path+'/splits/'+model+method+'_train_index_'+str(i)+'_'+dataset_name+'.csv', train_index, delimiter=',')
    np.savetxt(current_path+'/splits/'+model+method+'_test_index_'+str(i)+'_'+dataset_name+'.csv', test_index, delimiter=',')
    
    #pipeline = get_gbdt_pipeline(tcga=tcga,n_iter=n_iter, model=model)
    rs.fit(X_train, y_train, estimator__eval_test_size=validation_size, estimator__verbose=0)
    best_preds_train = rs.best_estimator_.predict(X_train)
    best_preds_test = rs.best_estimator_.predict(X_test)
    np.savetxt(current_path+'/predictions/'+model+method+'_best_preds_train_'+str(i)+'_'+dataset_name+'.csv', best_preds_train, delimiter=',')
    np.savetxt(current_path+'/predictions/'+model+method+'_best_preds_test_'+str(i)+'_'+dataset_name+'.csv', best_preds_test, delimiter=',')


    # save hyperparameter settings
    params = rs.best_estimator_.get_params()
    params_df = pd.DataFrame([params])

    # Save the DataFrame to a CSV file
    params_df.to_csv(current_path+'/params/'+model+method+'_best_params_'+str(i)+'_'+dataset_name+'.csv', index=False)

    if model in ['cind']:
            cindex_score_train = cindex_censored(y_train, -best_preds_train)
            print('Concordance Index',cindex_score_train)
            ibs_score_train = np.nan
            cindex_score_test = cindex_censored(y_test,-best_preds_test)
            print('Concordance Index Test',cindex_score_test)
            ibs_score_test = np.nan
    else:
        if model in ['eh']:
            durations_train, events_train = transform_back(y_train.values[:,0])
            durations_test, events_test = transform_back(y_test.values[:,0])
        else:
            durations_train, events_train = transform_back(y_train.values)
            durations_test, events_test = transform_back(y_test.values)
            
        cum_hazard_train = cum_hazard_dict[model](
                X_train.values, X_train.values, y_train.values, y_train.values,
                best_preds_train, best_preds_train
                )

        df_survival_train = np.exp(-cum_hazard_train)
        #durations_train, events_train = transform_back(y_train.values)
        time_grid_train = np.linspace(durations_train.min(), durations_train.max(), 100)
        ev = EvalSurv(df_survival_train, durations_train, events_train, censor_surv='km')
        cindex_score_train = ev.concordance_td('antolini')
        ibs_score_train = ev.integrated_brier_score(time_grid_train)
        

        cum_hazard_test = cum_hazard_dict[model](
                X_train.values, X_test.values, y_train.values, y_test.values,
                best_preds_train, best_preds_test
                )
        df_survival_test = np.exp(-cum_hazard_test)
        #durations_test, events_test = transform_back(y_test.values)
        time_grid_test = np.linspace(durations_test.min(), durations_test.max(), 100)
        ev = EvalSurv(df_survival_test, durations_test, events_test, censor_surv='km')
        cindex_score_test = ev.concordance_td('antolini')
        ibs_score_test = ev.integrated_brier_score(time_grid_test)
        print('Concordance Index',cindex_score_test)
        print('Integrated Brier Score:',ibs_score_test)
    metric = {'model':model, 'dataset':dataset_name, 'cindex_train':[cindex_score_train], 'cindex_test':[cindex_score_test], 'ibs_train':[ibs_score_train], 'ibs_score_test':[ibs_score_test]}
    pd.DataFrame(metric).to_csv(current_path+'/metrics/'+model+method+'_metric_'+str(i)+'_'+dataset_name+'.csv', index=False)
    return metric




# old


def get_gbdt_pipeline(
        tcga=False, 
        model='breslow', 
        rand_state=42, 
        early_stopping_rounds=10, 
        base_score=0.0, 
        #validation_size=0.2,
        verbose=1, 
        one_hot=False, 
        n_iter=50, 
        n_inner_splits=5
        ):
    # model options: breslow, efron, eh, aft, ah, cboost
    # read params from pickle
    np.random.seed(rand_state)
    if tcga==True:
        with open(pickle_path+model+'_gbdt_tcga.pkl', 'rb') as file:
            params = pickle.load(file)
    else:
        with open(pickle_path+model+'_gbdt_not_tcga.pkl', 'rb') as file:
            params = pickle.load(file)
    # column transformer
    if one_hot:
        ct = make_column_transformer(
            (StandardScaler(), make_column_selector(dtype_include=['float32'])),
            (OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist'), make_column_selector(dtype_include=['category', 'object'])),
            remainder='passthrough')
    else:
        ct = make_column_transformer(
            (StandardScaler(), make_column_selector(dtype_include=['float32'])),
            remainder='passthrough')
        
    # define scoring function
    def custom_scoring_function(y_true, y_pred, model):

        if not isinstance(y_true, np.ndarray):
            y_true = y_true.values
        if not isinstance(y_pred, np.ndarray):
            y_pred = y_pred.values
        #print('(y_true, y_pred)',y_true, y_pred)
        if np.any(y_pred == 0):
            print("The array contains zeros.")
        score = loss_dict[model](y_true, y_pred)
        print('score', score)
        return score

    scoring_function = make_scorer(custom_scoring_function,model=model, greater_is_better=False)

    # custom splitter
    inner_custom_cv = CustomSplit(n_splits=n_inner_splits, shuffle=True, random_state=rand_state)

    # set XGBSurv model
    estimator = XGBSurv(
        objective='breslow_objective',
        eval_metric='breslow_loss',
        random_state=rand_state, 
        disable_default_eval_metric=True,
        early_stopping_rounds=early_stopping_rounds, 
        base_score=base_score,
                    )
    print('estimator', estimator)
    pl = Pipeline([('scaler',ct),
                    ('estimator', estimator)])
        
    rs = RandomizedSearchCV(pl, params, scoring = scoring_function, n_jobs=-1, 
                                cv=inner_custom_cv, n_iter=n_iter, refit=True, 
                                random_state=rand_state, verbose=verbose,
                                error_score = 'raise')


    return rs


#define train function that just takes the split indices and then does the training saves everything and returns the cindex and ibs scores

def train_gbdt(X, 
               y, 
               i, 
               pipeline, 
               train_index, 
               test_index, 
               model, 
               dataset_name, 
               validation_size, 
               tcga=False,
               n_iter=50,
               ):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    X_train, y_train = sort_X_y_pandas(X_train, y_train)
    X_test, y_test = sort_X_y_pandas(X_test, y_test)
    #print(X_train,y_train)

    # save splits and data 
    np.savetxt(current_path+'/splits/'+model+'_train_index_'+str(i)+'_'+dataset_name+'.csv', train_index, delimiter=',')
    np.savetxt(current_path+'/splits/'+model+'_test_index_'+str(i)+'_'+dataset_name+'.csv', test_index, delimiter=',')
    
    pipeline = get_gbdt_pipeline(tcga=tcga,n_iter=n_iter, model=model)
    pipeline.fit(X_train, y_train, estimator__eval_test_size=validation_size, estimator__verbose=0)
    best_preds_train = pipeline.best_estimator_.predict(X_train)
    best_preds_test = pipeline.best_estimator_.predict(X_test)
    np.savetxt(current_path+'/predictions/'+model+'_best_preds_train_'+str(i)+'_'+dataset_name+'.csv', best_preds_train, delimiter=',')
    np.savetxt(current_path+'/predictions/'+model+'_best_preds_test_'+str(i)+'_'+dataset_name+'.csv', best_preds_test, delimiter=',')


    # save hyperparameter settings
    params = pipeline.best_estimator_.get_params()
    params_df = pd.DataFrame([params])

    # Save the DataFrame to a CSV file
    params_df.to_csv(current_path+'/params/'+model+'_best_params_'+str(i)+'_'+dataset_name+'.csv', index=False)

    cum_hazard_train = cum_hazard_dict[model](
            X_train.values, X_train.values, y_train.values, y_train.values,
            best_preds_train.reshape(-1), best_preds_train.reshape(-1)
            )

    df_survival_train = np.exp(-cum_hazard_train)
    # remember eh etc. only report test
    if model in 'eh':
        durations_test, events_test = transform_back(y_train.values[:,0])
        durations_test, events_test = transform_back(y_test.values[:,0])
    else:
        durations_train, events_train = transform_back(y_train.values)
        durations_test, events_test = transform_back(y_test.values)
        
    time_grid_train = np.linspace(durations_train.min(), durations_train.max(), 100)
    ev = EvalSurv(df_survival_train, durations_train, events_train, censor_surv='km')
    cindex_score_train = ev.concordance_td('antolini')
    ibs_score_train = ev.integrated_brier_score(time_grid_train)
    

    cum_hazard_test = cum_hazard_dict[model](
            X_train.values, X_test.values, y_train.values, y_test.values,
            best_preds_train.reshape(-1), best_preds_test.reshape(-1)
            )
    df_survival_test = np.exp(-cum_hazard_test)

    time_grid_test = np.linspace(durations_test.min(), durations_test.max(), 100)
    ev = EvalSurv(df_survival_test, durations_test, events_test, censor_surv='km')
    cindex_score_test = ev.concordance_td('antolini')
    ibs_score_test = ev.integrated_brier_score(time_grid_test)
    print('Concordance Index',cindex_score_test)
    print('Integrated Brier Score:',ibs_score_test)
    metric = {'model':model, 'dataset':dataset_name, 'cindex_train':[cindex_score_train], 'cindex_test':[cindex_score_test], 'ibs_train':[ibs_score_train], 'ibs_score_test':[ibs_score_test]}
    pd.DataFrame(metric).to_csv(current_path+'/metrics/'+model+'_metric_'+str(i)+'_'+dataset_name+'.csv', index=False)
    return metric