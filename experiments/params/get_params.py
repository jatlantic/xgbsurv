
import pickle
from scipy.stats import loguniform, randint, uniform
import os

# get path

current_path = os.getcwd()  # Get the current working directory path
current_path = current_path+'/Documents/xgbsurv/experiments/params/'

# gbdts

def make_param_grid_gbdt(learning_rate, n_estimators):
    param_grid = param_grid = {
    'estimator__reg_alpha': loguniform(1e-10,1),#[1e-10,1], # from hyp augmentation, L1 regularization
    'estimator__reg_lambda': loguniform(1e-10,1), #[1e-10,1], #alias l2_regularization, lambda in augmentation
    'estimator__learning_rate': loguniform(learning_rate[0],learning_rate[1]), #[0.001,1], # assumed alias eta from augmentation,
    'estimator__n_estimators':  randint(n_estimators[0],n_estimators[1]),#00), # corresponds to num_rounds
    'estimator__gamma': loguniform(0.001,1.0),#[0.1,1], # minimum loss reduction required to make a further partition on a leaf node of the tree.
    'estimator__colsample_bylevel': uniform(0.1, 1-0.1), #[0.1,1], # from hyp augmentation
    'estimator__colsample_bynode': uniform(0.1, 1-0.1), #[0.1,1], # from hyp augmentation, uniform(0.1,1),
    'estimator__colsample_bytree': uniform(0.5, 1-0.5),#[0.5,1], # from hyp augmentation, seems to exceed the bound, uniform(0.5,1)
    'estimator__max_depth': randint(1,20),#[1,20], # from hyp augmentation
    'estimator__max_delta_step': randint(0,10),#[0,10], # from hyp augmentation
    'estimator__min_child_weight' : loguniform(0.1,20-0.1),#[0.1,20], # from hyp augmentation
    'estimator__subsample': uniform(0.01,1-0.01),#[0.01,1], # from hyp augmentation
    }
    return param_grid


scenarios = {
    'breslow_gbdt_not_tcga': {'learning_rate': (0.01, 1.0), 'n_estimators': (1, 4000)},
    'breslow_gbdt_tcga': {'learning_rate': (0.01, 1.0), 'n_estimators': (1, 4000)},
    'efron_gbdt_not_tcga': {'learning_rate': (0.01, 1.0), 'n_estimators': (1, 4000)},
    'efron_gbdt_tcga': {'learning_rate': (0.01, 1.0), 'n_estimators': (1, 4000)},
    'aft_gbdt_not_tcga': {'learning_rate': (0.01, 1.0), 'n_estimators': (1, 4000)},
    'aft_gbdt_tcga': {'learning_rate': (0.01, 1.0), 'n_estimators': (1, 4000)},
    'eh_gbdt_not_tcga': {'learning_rate': (0.01, 1.0), 'n_estimators': (1, 4000)},
    'eh_gbdt_tcga': {'learning_rate': (0.01, 1.0), 'n_estimators': (1, 4000)},
    'ah_gbdt_not_tcga': {'learning_rate': (0.01, 1.0), 'n_estimators': (1, 4000)},
    'ah_gbdt_tcga': {'learning_rate': (0.01, 1.0), 'n_estimators': (1, 4000)},
    'cind_gbdt_not_tcga': {'learning_rate': (0.01, 1.0), 'n_estimators': (1, 4000)},
    'cind_gbdt_tcga': {'learning_rate': (0.01, 1.0), 'n_estimators': (1, 4000)}
}

# Save each param_grid with its respective name
for name, params in scenarios.items():
    param_grid = make_param_grid_gbdt(params['learning_rate'], params['n_estimators'])
    with open(current_path+name+'.pkl', 'wb') as f:
        pickle.dump(param_grid, f)


# deep learning

def make_param_grid_dl(learning_rate, max_epochs):
    param_grid = {
    'estimator__module__n_layers': [1, 2, 4],
    'estimator__module__num_nodes': [64, 128, 256, 512],
    'estimator__module__dropout': uniform(0.0,0.7),
    'estimator__optimizer__weight_decay': [0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0],
    'estimator__batch_size': [64, 128, 256, 512, 1024],
    'estimator__lr':[learning_rate], #0.01
    'estimator__max_epochs':  randint(max_epochs[0], max_epochs[1]) 
}
    return param_grid



scenarios = {
    'breslow_dl_not_tcga': {'learning_rate': 0.01, 'max_epochs': (150, 250)},
    'breslow_dl_tcga': {'learning_rate': 0.01, 'max_epochs': (150, 250)},
    'efron_dl_not_tcga': {'learning_rate': 0.01, 'max_epochs': (150, 250)},
    'efron_dl_tcga': {'learning_rate': 0.01, 'max_epochs': (150, 250)},
    'aft_dl_not_tcga': {'learning_rate': 0.01, 'max_epochs': (150, 250)},
    'aft_dl_tcga': {'learning_rate': 0.01, 'max_epochs': (150, 250)},
    'eh_dl_not_tcga': {'learning_rate': 0.0001, 'max_epochs': (10, 20)},
    'eh_dl_tcga': {'learning_rate': 0.0001, 'max_epochs': (150, 250)},
    'ah_dl_not_tcga': {'learning_rate': 0.00001, 'max_epochs': (10, 15)},
    'ah_dl_tcga': {'learning_rate': 0.001, 'max_epochs': (10, 20)},
    'cind_dl_not_tcga': {'learning_rate': 0.01, 'max_epochs': (150, 250)},
    'cind_dl_tcga': {'learning_rate': 0.01, 'max_epochs': (150, 250)}
}

# Save each param_grid with its respective name
for name, params in scenarios.items():
    param_grid = make_param_grid_dl(params['learning_rate'], params['max_epochs'])
    with open(current_path+name+'.pkl', 'wb') as f:
        pickle.dump(param_grid, f)