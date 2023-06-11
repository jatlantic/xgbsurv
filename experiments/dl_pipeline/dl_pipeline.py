# gbdt pipeline and model
import pickle
import pandas as pd
import numpy as np
from numpy import savetxt
from xgbsurv.datasets import (load_metabric, load_flchain, load_rgbsg, load_support, load_tcga)
from xgbsurv.models.utils import sort_X_y_pandas, transform_back, transform
from xgbsurv.models.breslow_final import get_cumulative_hazard_function_breslow, breslow_estimator_loop
import torch
from skorch.callbacks import GradientNormClipping
from torch import nn
from xgbsurv.evaluation import cindex_censored
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.decomposition import PCA
from skorch import NeuralNet
from skorch.callbacks import EarlyStopping, Callback, LRScheduler
import skorch.callbacks
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import check_cv
from numbers import Number
import torch.utils.data
from skorch.utils import flatten
from skorch.utils import is_pandas_ndframe
from skorch.utils import check_indexing
from skorch.utils import multi_indexing
from skorch.utils import to_numpy
from skorch.dataset import get_len
from skorch.dataset import ValidSplit
from pycox.evaluation import EvalSurv
from scipy.stats import uniform as scuniform
from scipy.stats import randint as scrandint
from scipy.stats import loguniform as scloguniform
import random
import os
import sys
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

method = '_dl_'
rand_state = 42
# path
current_path = os.getcwd()  # Get the current working directory path
#pickle_path = current_path+'/Documents/xgbsurv/experiments/params/'
somepath = os.path.abspath(os.path.join(current_path,  ".."))
pickle_path = somepath+'/params/'

sys.path.append(somepath+'/deep_learning/')
from loss_functions_pytorch import BreslowLoss, breslow_likelihood_torch, \
EfronLoss, efron_likelihood_torch, EHLoss, eh_likelihood_torch_2, \
aft_likelihood_torch, AFTLoss, ah_likelihood_torch, AHLoss, \
cind_likelihood_torch, CindLoss

loss_dict = {
            'breslow': breslow_likelihood_torch, 
            'efron': efron_likelihood_torch, 
            'cind': cind_likelihood_torch, 
            #'deephit':deephit_loss1_pycox, 
            'aft': aft_likelihood_torch, 
            'ah': ah_likelihood_torch, 
            'eh': eh_likelihood_torch_2
            }

Loss_dict = {
            'breslow': BreslowLoss, 
            'efron':  EfronLoss, 
            'cind': CindLoss, 
            #'deephit':deephit_loss1_pycox, 
            'aft': AFTLoss, 
            'ah': AHLoss, 
            'eh': EHLoss
            }

cum_hazard_dict ={
            'breslow': get_cumulative_hazard_function_breslow, 
            'efron': get_cumulative_hazard_function_efron, 
            #'cind': cind_loss, 
            #'deephit':deephit_loss1_pycox, 
            'aft': get_cumulative_hazard_function_aft, 
            'ah':get_cumulative_hazard_function_ah, 
            'eh': get_cumulative_hazard_function_eh
                }

# custom splitter



def seed_torch(seed=rand_state):
    """Sets all seeds within torch and adjacent libraries.

    Args:
        seed: Random seed to be used by the seeding functions.

    Returns:
        None
    """
    random.seed(seed)
    #os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    return None

class FixSeed(Callback):
    def __init__(self, seed):
        self.seed = seed

    def initialize(self):
        seed_torch(self.seed)
        return super().initialize()
    
class SurvivalModel(nn.Module):
    def __init__(self, n_layers, input_units, num_nodes, dropout, out_features):
        super(SurvivalModel, self).__init__()
        self.n_layers = n_layers
        self.in_features = input_units
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.out_features = out_features
        model = []
        # first layer
        model.append(torch.nn.Linear(input_units, num_nodes))
        model.append(torch.nn.ReLU())
        model.append(torch.nn.Dropout(dropout))
        model.append(torch.nn.BatchNorm1d(num_nodes))

        for i in range(n_layers-1):
            model.append(torch.nn.Linear(num_nodes, num_nodes))
            #init.kaiming_normal_(model[-1].weight, nonlinearity='relu')
            model.append(torch.nn.ReLU())
            model.append(torch.nn.Dropout(dropout))
            model.append(torch.nn.BatchNorm1d(num_nodes))

        # output layer
        model.append(torch.nn.Linear(num_nodes, out_features))
    
        self.layers = nn.Sequential(*model)

    def forward(self, X):
        X = X.to(torch.float32)
        res = self.layers(X)
        #print(res)
        return res


    
class CustomValidSplit():

    def __init__(
            self,
            cv=5,
            stratified=False,
            random_state=None,
    ):
        self.stratified = stratified
        self.random_state = random_state

        if isinstance(cv, Number) and (cv <= 0):
            raise ValueError("Numbers less than 0 are not allowed for cv "
                             "but ValidSplit got {}".format(cv))

        if not self._is_float(cv) and random_state is not None:
            raise ValueError(
                "Setting a random_state has no effect since cv is not a float. "
                "You should leave random_state to its default (None), or set cv "
                "to a float value.",
            )

        self.cv = cv

    def _is_stratified(self, cv):
        return isinstance(cv, (StratifiedKFold, StratifiedShuffleSplit))

    def _is_float(self, x):
        if not isinstance(x, Number):
            return False
        return not float(x).is_integer()

    def _check_cv_float(self):
        cv_cls = StratifiedShuffleSplit if self.stratified else ShuffleSplit
        return cv_cls(test_size=self.cv, random_state=self.random_state)

    def _check_cv_non_float(self, y):
        return check_cv(
            self.cv,
            y=y,
            classifier=self.stratified,
        )

    def check_cv(self, y):
        """Resolve which cross validation strategy is used."""
        y_arr = None
        if self.stratified:
            # Try to convert y to numpy for sklearn's check_cv; if conversion
            # doesn't work, still try.
            try:
                y_arr = to_numpy(y)
            except (AttributeError, TypeError):
                y_arr = y

        if self._is_float(self.cv):
            return self._check_cv_float()
        return self._check_cv_non_float(y_arr)

    def _is_regular(self, x):
        return (x is None) or isinstance(x, np.ndarray) or is_pandas_ndframe(x)

    def __call__(self, dataset, y=None, groups=None):
        # key change here
        y = np.sign(y)
        bad_y_error = ValueError(
            "Stratified CV requires explicitly passing a suitable y.")
        if (y is None) and self.stratified:
            raise bad_y_error

        cv = self.check_cv(y)
        if self.stratified and not self._is_stratified(cv):
            raise bad_y_error

        # pylint: disable=invalid-name
        len_dataset = get_len(dataset)
        if y is not None:
            len_y = get_len(y)
            if len_dataset != len_y:
                raise ValueError("Cannot perform a CV split if dataset and y "
                                 "have different lengths.")

        args = (np.arange(len_dataset),)
        if self._is_stratified(cv):
            args = args + (to_numpy(y),)

        idx_train, idx_valid = next(iter(cv.split(*args, groups=groups)))
        dataset_train = torch.utils.data.Subset(dataset, idx_train)
        dataset_valid = torch.utils.data.Subset(dataset, idx_valid)
        return dataset_train, dataset_valid

class InputShapeSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        net.set_params(module__input_units=X.shape[-1])



def train_dl_complete(
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
    print('model:', model)
    # model options: breslow, efron, eh, aft, ah, cboost
    # read params from pickle
    np.random.seed(rand_state)
    if tcga==True:
        with open(pickle_path+model+'_dl_tcga.pkl', 'rb') as file:
            params = pickle.load(file)
    else:
        with open(pickle_path+model+'_dl_not_tcga.pkl', 'rb') as file:
            params = pickle.load(file)
    # column transformer
    if one_hot:
        ct = make_column_transformer(
            (StandardScaler(), make_column_selector(dtype_include=['float32'])),
            (OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist'), make_column_selector(dtype_include=['category', 'object'])),
            remainder='passthrough')
    else:
        ct = make_column_transformer(
            (StandardScaler(), make_column_selector(dtype_include=['float32'],dtype_exclude=['category', 'object'])),
            remainder='passthrough')
        
    
        
    # define scoring function
    def custom_scoring_function(y_true, y_pred, model):

        #y_true = torch.from_numpy(y_true)
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)
        if isinstance(y_pred, pd.Series):
            y_pred = torch.tensor(y_pred.values)
        if isinstance(y_true, pd.Series):
            y_true = torch.tensor(y_true.values)
        #print('y_true',y_true)
        #print('y_pred',y_pred)
        score = loss_dict[model](y_pred, y_true) #.to(torch.float32)
        return score.numpy()

    scoring_function = make_scorer(custom_scoring_function, model=model, greater_is_better=False)
    
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
            #print(bins)
            return super().split(X, bins, groups=groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
        


    # custom splitter
    inner_custom_cv = CustomSplit(n_splits=n_inner_splits, shuffle=True, random_state=rand_state)

    if model in ['eh']:
        out_features = 2
    else:
        out_features = 1

    # set XGBSurv model
    estimator = NeuralNet(
        SurvivalModel, 
        module__input_units = X.shape[1],
        module__out_features = out_features,
        # for split sizes when result size = 1
        iterator_train__drop_last=True,
        #iterator_valid__drop_last=True,
        criterion=Loss_dict[model],
        optimizer=torch.optim.AdamW,
        optimizer__weight_decay = 0.4,
        batch_size=32, # separate train and valid->iterator_train__batch_size=128 and iterator_valid__batch_size=128 ?
        callbacks=[
            (
                "sched",
                LRScheduler(
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                    monitor="valid_loss",
                    patience=5,
                ),
            ),
            (
                "es",
                EarlyStopping(
                    monitor="valid_loss",
                    patience=early_stopping_rounds,
                    load_best=True,
                    threshold=0.01, #changed for ah
                ),
            ),
            ("seed", FixSeed(seed=42)),
            ("Input Shape Setter",InputShapeSetter())
        ],
        train_split = CustomValidSplit(0.2, stratified=True, random_state=rand_state), 
        verbose=0
    )
    if model in ['ah']:
        print('gradient clipping')
        estimator.callbacks.append(("gradientclip",GradientNormClipping(gradient_clip_value=0.05,gradient_clip_norm_type=1)))

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
    np.savetxt(current_path+'/splits/'+model+method+'train_index_'+str(i)+'_'+dataset_name+'.csv', train_index, delimiter=',')
    np.savetxt(current_path+'/splits/'+model+method+'test_index_'+str(i)+'_'+dataset_name+'.csv', test_index, delimiter=',')
    
    #pipeline = get_gbdt_pipeline(tcga=tcga,n_iter=n_iter, model=model)
    #if model in ['eh']:
    #    X_train = X_train.to_numpy()
    #    y_train = y_train.to_numpy()
    rs.fit(X_train, y_train)
    best_preds_train = rs.best_estimator_.predict(X_train)
    best_preds_test = rs.best_estimator_.predict(X_test)
    np.savetxt(current_path+'/predictions/'+model+method+'best_preds_train_'+str(i)+'_'+dataset_name+'.csv', best_preds_train, delimiter=',')
    np.savetxt(current_path+'/predictions/'+model+method+'best_preds_test_'+str(i)+'_'+dataset_name+'.csv', best_preds_test, delimiter=',')


    # save hyperparameter settings
    params = rs.best_estimator_.get_params()
    params_df = pd.DataFrame([params])

    # Save the DataFrame to a CSV file
    params_df.to_csv(current_path+'/params/'+model+method+'best_params_'+str(i)+'_'+dataset_name+'.csv', index=False)

    if model in ['cind']:
            cindex_score_train = cindex_censored(y_train, -best_preds_train.reshape(-1))
            print('Concordance Index',cindex_score_train)
            ibs_score_train = np.nan
            cindex_score_test = cindex_censored(y_test,-best_preds_test.reshape(-1))
            print('Concordance Index Test',cindex_score_test)
            ibs_score_test = np.nan
    else:
        if model in ['eh']:
            durations_train, events_train = transform_back(y_train.values)
            durations_test, events_test = transform_back(y_test.values)
        
        else:
            durations_train, events_train = transform_back(y_train.values)
            durations_test, events_test = transform_back(y_test.values)
            best_preds_train = best_preds_train.reshape(-1)
            best_preds_test = best_preds_test.reshape(-1)

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
    metric = {'model':model, 'dataset':dataset_name, 'cindex_train':[cindex_score_train], 'cindex_test':[cindex_score_test], 'ibs_train':[ibs_score_train], 'ibs_test':[ibs_score_test]}
    pd.DataFrame(metric).to_csv(current_path+'/metrics/'+model+'_metric_'+str(i)+'_'+dataset_name+'.csv', index=False)
    return metric

