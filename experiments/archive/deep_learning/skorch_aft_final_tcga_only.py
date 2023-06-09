import sys
sys.path.append('/Users/JUSC/Documents/xgbsurv/experiments/deep_learning')
import pandas as pd
import numpy as np
from numpy import savetxt
from xgbsurv.datasets import (load_metabric, load_flchain, load_rgbsg, load_support, load_tcga)
from xgbsurv.models.utils import sort_X_y_pandas, transform_back, transform
from xgbsurv.models.eh_aft_final import get_cumulative_hazard_function_aft
import torch
from torch import nn
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.decomposition import PCA
from loss_functions_pytorch import AFTLoss, aft_likelihood_torch
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
#torch.set_default_dtype(torch.float64)
#torch.set_default_tensor_type(torch.DoubleTensor)

path = '/Users/JUSC/Documents/xgbsurv/experiments/deep_learning/'
n_outer_splits = 5
n_inner_splits = 5
rand_state = 42
n_iter = 50
early_stopping_rounds=10
base_score = 0.0

# set seed for scipy/numpy
np.random.seed(rand_state)

param_grid_aft = {
    'estimator__module__n_layers': [1, 2, 4],
    'estimator__module__num_nodes': [64, 128, 256, 512],
    'estimator__module__dropout': scuniform(0.0,0.7),
    'estimator__optimizer__weight_decay': [0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0],
    'estimator__batch_size': [64, 128, 256, 512, 1024],
    #lr not in paper because of learning rate finder
    # note: setting learning rate higher would make exp(partial_hazard) explode
    #'estimator__lr': scloguniform(0.001,0.01), # scheduler unten einbauen
    # use callback instead
    'estimator__lr':[0.01]
    #'max_epochs':  scrandint(10,20), # corresponds to num_rounds
}


def seed_torch(seed=42):
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

# Define Scorer
def custom_scoring_function(y_true, y_pred):

        #y_true = torch.from_numpy(y_true)
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)
        if isinstance(y_pred, pd.Series):
            y_pred = torch.tensor(y_pred.values)
        if isinstance(y_true, pd.Series):
            y_true = torch.tensor(y_true.values)
        #print('loss function y_pred', y_pred)
        #print('loss function y_true', y_true)
        score = -aft_likelihood_torch(y_pred, y_true) #.to(torch.float32)
        return score.numpy()

# maybe in skorch the loss funciton lower is better is reversed
# in terms of performance this seems to work
scoring_function = make_scorer(custom_scoring_function, greater_is_better=False)



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

        # for layer in self.layers:
        #     if isinstance(layer, nn.Linear):
        #         #nn.init.uniform_(layer.weight, a=-0.5, b=0.5)
        #         nn.init.kaiming_normal_(layer.weight)


    def forward(self, X):
        X = X.to(torch.float32)
        res = self.layers(X)
        #print(res)
        return res


class CustomStandardScaler(StandardScaler):
    
    def __init__(self, copy=True, with_mean=True, with_std=True):
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)
        
    def fit(self, X, y=None):
        return super().fit(X, y)
    
    def transform(self, X, y=None):
        X_transformed = super().transform(X, y)
        return X_transformed.astype(np.float32)
    
    def fit_transform(self, X, y=None):
        X_transformed = super().fit_transform(X, y)
        return X_transformed.astype(np.float32)

# Define stratified inner k-fold cross-validation
class CustomSplit(StratifiedKFold):
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y, groups=None):
        print('split', X.dtypes)
        try:
            if y.shape[1]>1:
                y = y[:,0]
        except:
            pass
        bins = np.sign(y)
        return super().split(X, bins, groups=groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

outer_custom_cv = CustomSplit(n_splits=n_outer_splits, shuffle=True, random_state=rand_state)
inner_custom_cv = CustomSplit(n_splits=n_outer_splits, shuffle=True, random_state=rand_state)






class CustomStandardScaler(StandardScaler):
    
    def __init__(self, copy=True, with_mean=True, with_std=True):
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)
        
    def fit(self, X, y=None):
        return super().fit(X, y)
    
    def transform(self, X, y=None):
        X_transformed = super().transform(X, y)
        return X_transformed.astype(np.float32)
    
    def fit_transform(self, X, y=None):
        X_transformed = super().fit_transform(X, y)
        return X_transformed.astype(np.float32)
    
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


# ## TCGA

param_grid_aft_tcga = {
    'estimator__module__n_layers': [1, 2, 4],
    'estimator__module__num_nodes': [64, 128, 256, 512],
    'estimator__module__dropout': scuniform(0.0,0.7),
    'estimator__optimizer__weight_decay': [0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0],
    'estimator__batch_size': [64, 128, 256, 512, 1024],
    #lr not in paper because of learning rate finder
    # note: setting learning rate higher would make exp(partial_hazard) explode
    #'estimator__lr': scloguniform(0.001,0.01), # scheduler unten einbauen
    # use callback instead
    'estimator__lr':[0.01],
    #'estimator__max_epochs':  scrandint(10,20), # corresponds to num_rounds
    #'pca__n_components': [8, 16, 32, 64]
}


def train_eval(X, y, net, n_iter, filename):
        model = 'aft_'
        dataset_name = filename.split('_')[0]
        # add IBS later
        outer_scores = {'cindex_test_'+dataset_name:[], 'ibs_test_'+dataset_name:[]}
        best_params = {'best_params_'+dataset_name:[]}
        best_model = {'best_model_'+dataset_name:[]}
        ct = make_column_transformer(
                #(OneHotEncoder(sparse_output=False), make_column_selector(dtype_include=['category', 'object']))
                (StandardScaler(), make_column_selector(dtype_include=['float32']))
                ,remainder='drop')
        pipe = Pipeline([('scaler',ct),
                         #('pca', PCA()),#n_components=10
                        ('estimator', net)])
        rs = RandomizedSearchCV(pipe, param_grid_aft_tcga, scoring = scoring_function, n_jobs=-1, 
                                    n_iter=n_iter, refit=True)
        for i, (train_index, test_index) in enumerate(outer_custom_cv.split(X, y)):
                # Split data into training and testing sets for outer fold
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                X_train, y_train = sort_X_y_pandas(X_train, y_train)
                X_test, y_test = sort_X_y_pandas(X_test, y_test)

                #print(X_train.shape, type(X_train))
                #print(y_train.shape, type(y_train))
                #print(X_test.shape, type(X_test))
                #print(y_test.shape, type(y_test))
                # save splits and data
                savetxt(path+'splits/train_index_'+str(i)+'_'+filename, train_index, delimiter=',')
                savetxt(path+'splits/test_index_'+str(i)+'_'+filename, test_index, delimiter=',')
                
                # savetxt('splits/X_train_'+str(i)+'_'+filename, X_train, delimiter=',')
                # savetxt('splits/X_test_'+str(i)+'_'+filename, X_test, delimiter=',')

                # savetxt('splits/y_train_'+str(i)+'_'+filename, y_train, delimiter=',')
                # savetxt('splits/y_test_'+str(i)+'_'+filename, y_test, delimiter=',')

                rs.fit(X_train, y_train)
                best_preds_train = rs.best_estimator_.predict(X_train)
                best_preds_test = rs.best_estimator_.predict(X_test)
                savetxt(path+'predictions/'+model+'best_preds_train_'+str(i)+'_'+filename, best_preds_train, delimiter=',')
                savetxt(path+'predictions/'+model+'best_preds_test_'+str(i)+'_'+filename, best_preds_test, delimiter=',')

                
                # save hyperparameter settings
                params = rs.best_estimator_.get_params
                best_params['best_params_'+dataset_name] += [rs.best_params_]
                best_model['best_model_'+dataset_name] += [params]
                    
                try:
                    cum_hazard_test = get_cumulative_hazard_function_aft(
                            X_train.values, X_test.values, y_train.values, y_test.values,
                            best_preds_train.reshape(-1), best_preds_test.reshape(-1)
                            )
                    df_survival_test = np.exp(-cum_hazard_test)
                    durations_test, events_test = transform_back(y_test.values)
                    time_grid_test = np.linspace(durations_test.min(), durations_test.max(), 100)
                    ev = EvalSurv(df_survival_test, durations_test, events_test, censor_surv='km')
                    print('Concordance Index',ev.concordance_td('antolini'))
                    print('Integrated Brier Score:',ev.integrated_brier_score(time_grid_test))
                    cindex_score_test = ev.concordance_td('antolini')
                    ibs_score_test = ev.integrated_brier_score(time_grid_test)

                    outer_scores['cindex_test_'+dataset_name] += [cindex_score_test]
                    outer_scores['ibs_test_'+dataset_name] += [ibs_score_test]
                except: 
                    outer_scores['cindex_test_'+dataset_name] += [np.nan]
                    outer_scores['ibs_test_'+dataset_name] += [np.nan]
            
        df_best_params = pd.DataFrame(best_params)
        df_best_model = pd.DataFrame(best_model)
        df_outer_scores = pd.DataFrame(outer_scores)
        df_metrics = pd.concat([df_best_params,df_best_model,df_outer_scores], axis=1)
        df_metrics.to_csv(path+'metrics/'+model+'metric_summary'+str(i)+'_'+filename, index=False)
        # cindex
        df_agg_metrics_cindex = pd.DataFrame({'dataset':[dataset_name],
                                              'cindex_test_mean':df_outer_scores['cindex_test_'+dataset_name].mean(),
                                              'cindex_test_std':df_outer_scores['cindex_test_'+dataset_name].std() })
        # IBS
        df_agg_metrics_ibs = pd.DataFrame({'dataset':[dataset_name],
                                              'ibs_test_mean':df_outer_scores['ibs_test_'+dataset_name].mean(),
                                              'ibs_test_std':df_outer_scores['ibs_test_'+dataset_name].std() })
        
        return df_agg_metrics_cindex, df_agg_metrics_ibs, best_model, best_params, outer_scores, best_preds_train, best_preds_test 

 
cancer_types = [
    'BLCA',
    'BRCA',
    'HNSC',
    'KIRC',
    'LGG',
    'LIHC',
    'LUAD',
    'LUSC',
    'OV',
    'STAD'
    ]
    
agg_metrics_cindex = []
agg_metrics_ibs = []

class InputShapeSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        net.set_params(module__input_units=X.shape[-1])

for idx, cancer_type in enumerate(cancer_types):
    # get name of current dataset
    data = load_tcga(path="/Users/JUSC/Documents/xgbsurv/xgbsurv/datasets/data/", cancer_type=cancer_type, as_frame=True)
    X  = data.data #.astype(np.float32)
    y = data.target #.values #.to_numpy()

    X, y = sort_X_y_pandas(X, y)

    net = NeuralNet(
        SurvivalModel, 
        module__n_layers = 1,
        module__input_units = X.shape[1],
        #module__num_nodes = 32,
        #module__dropout = 0.1, # these could also be removed
        module__out_features = 1,
        # for split sizes when result size = 1
        iterator_train__drop_last=True,
        #iterator_valid__drop_last=True,
        criterion=AFTLoss,
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
                ),
            ),
            ("seed", FixSeed(seed=42)),
            #("Input Shape Setter",InputShapeSetter())
        ],#[EarlyStopping(patience=10)],#,InputShapeSetter()],
        #TODO: enable stratification, verify
        train_split = CustomValidSplit(0.2, stratified=True, random_state=rand_state), 
        #max_epochs=1, #0,#100
        #train_split=None,
        verbose=1
    )
    df_agg_metrics_cindex, df_agg_metrics_ibs, best_model,params, outer_scores, best_preds_train, best_preds_test = train_eval(X, y, net, n_iter, data.filename)
    agg_metrics_cindex.append(df_agg_metrics_cindex)
    agg_metrics_ibs.append(df_agg_metrics_ibs)

print(agg_metrics_cindex)
print(agg_metrics_ibs)
# df_final_aft_1_ibs = pd.concat([df for df in agg_metrics_ibs]).round(4)
# df_final_aft_1_ibs.to_csv(path+'metrics/final_deep_learning_aft_tcga_ibs.csv', index=False)
# df_final_aft_1_ibs.to_csv('/Users/JUSC/Documents/644928e0fb7e147893e8ec15/05_thesis/tables/final_deep_learning_aft_tcga_ibs.csv', index=False) 
# df_final_aft_1_ibs


# df_final_aft_1_cindex = pd.concat([df for df in agg_metrics_cindex]).round(4)
# df_final_aft_1_cindex.to_csv(path+'metrics/final_deep_learning_aft_tcga_cindex.csv', index=False)
# df_final_aft_1_cindex.to_csv('/Users/JUSC/Documents/644928e0fb7e147893e8ec15/05_thesis/tables/final_deep_learning_aft_tcga_cindex.csv', index=False)  #
# df_final_aft_1_cindex


