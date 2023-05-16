import pandas as pd
import numpy as np
from xgbsurv.models.utils import KaplanMeier
import os
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

# original data is taken from pycox
# https://github.com/havakv/pycox/tree/master/pycox/datasets

def metabric_preprocess(path="add your path here"):
    filename="original_data/METABRIC_pycox_full.csv"
    df = pd.read_csv(path+filename)
    # name columns
    df.columns = ['MKI67', 'EGFR', 'PGR', 'ERBB2', 
                  'hormone_treatment', 
                  'radiotherapy', 
                  'chemotherapy', 
                  'ER_positive', 
                  'age', 
                  'time',
                  'event']
    # remove zero time observations
    df = df[df.time!=0]
    # sort data
    df.sort_values(by='time', ascending=True, inplace=True)
    # save data
    df.to_csv(path+"data/METABRIC_adapted.csv", index=False)
    return

def flchain_preprocess(path="add your path here"):
    filename="original_data/FLCHAIN_full.csv"
    df = pd.read_csv(path+filename)
    # drop death cause column
    df.drop('chapter', inplace=True, axis=1)
    # there is one column with NA values (17%)
    # we will fill this with the median
    df['creatinine'] = df['creatinine'].fillna(df['creatinine'].median())
    # make gender column 0,1, i.e. make data numerical
    df['sex'] = LabelEncoder().fit_transform(df['sex'])
    # name columns
    df.columns = [
        "age",
        "sex",
        "sample_yr",
        "kappa",
        "lambda",
        "flc_grp",
        "creatinine",
        "mgus",
        "time", 
        "event"
        ]
    # remove zero time observations
    df = df[df.time!=0]
    # sort data
    df.sort_values(by='time', ascending=True, inplace=True)
    # save data
    df.to_csv(path+"data/FLCHAIN_adapted.csv", index=False)
    return


def rgbsg_preprocess(path="add your path here"):
    filename="original_data/RGBSG_pycox_full.csv"
    df = pd.read_csv(path+filename)
    # drop death cause column
    # name columns
    df.columns = [
        "horm_treatment",
        "grade",
        "menopause",
        "age",
        "n_positive_nodes",
        "progesterone",
        "estrogene",
        "time",
        "event"
        ]
    # remove zero time observations
    df = df[df.time!=0]
    # sort data
    df.sort_values(by='time', ascending=True, inplace=True)
    # save data
    df.to_csv(path+"data/RGBSG_adapted.csv", index=False)
    return

def support_preprocess(path="add your path here"):
    filename="original_data/SUPPORT_pycox_full.csv"
    df = pd.read_csv(path+filename)
    # drop death cause column
    # name columns
    df.columns = [
        "age",
        "sex",
        "race",
        "n_comorbidities",	
        "diabetes",
        "dementia",
        "cancer",
        "blood_pressure",
        "heart_rate",
        "respiration_rate",
        "temperature",
        "white_blood_cell",
        "serum_sodium",
        "serum_creatinine",
        "time",
        "event"
        ]
    # remove zero time observations
    df = df[df.time!=0]
    # sort data
    df.sort_values(by='time', ascending=True, inplace=True)
    # save data
    df.to_csv(path+"data/SUPPORT_adapted.csv", index=False)
    return

#rgbsg_preprocess(path="")
#metabric_preprocess(path="")
#flchain_preprocess(path="")
#support_preprocess(path="")


def tcga_preprocess(path="add your path here"):
    cancer_names = ['BLCA',
    'BRCA',
    'HNSC',
    'KIRC',
    'LGG',
    'LIHC',
    'LUAD',
    'LUSC',
    'OV',
    'STAD']
    print(os.getcwd())
    for name in cancer_names:
        filename="original_data/TCGA/"+name+'.csv'
        df = pd.read_csv(path+filename)
        # drop patient id column
        df.drop(['patient_id'],axis=1, inplace=True)
        # remove zero time observations
        df = df[df.time!=0]
        # sort data
        df.sort_values(by='time', ascending=True, inplace=True)
        # save data
        df.to_csv(path+"data/"+name+"_adapted.csv", index=False)
    return

tcga_preprocess(path="/Users/JUSC/Documents/xgbsurv/xgbsurv/datasets/")

#/Users/JUSC/Documents/xgbsurv/xgbsurv/datasets/original_data/TCGA

# discretizer for Deephit
def discretizer_df(df, n_cuts=10, type = 'equidistant', min_time=0.0) -> pd.DataFrame:
    """Discretize dataframe along time axis.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    n_cuts : int, optional
        _description_, by default 10
    type : str, optional
        _description_, by default 'equidistant'
    min_time : float, optional
        _description_, by default 0.0

    Returns
    -------
    pd.DataFrame
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    
    References
    ----------
    .. [1] Kvamme, H. havakv/pycox. (2023).
    .. [2] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """
    # TODO: Add check for pd dataframe, adapt for data, target structure
    if 'time' and 'event' not in df.columns:
        raise ValueError('Required columns are not in dataframe.')
    elif df.time.min()==0:
        raise ValueError('Time zero cannot exist in the data.')
    elif type=='equidistant':
        df = df.sort_values(by='time', ascending=True)
        time, _ = df.time.to_numpy(), df.event.to_numpy()
        cuts = np.linspace(min_time,time.max(),num=n_cuts)
        idx = np.digitize(time, cuts, right=False) #-1
        df['time'] = idx
    elif type=='quantiles':
        df = df.sort_values(by='time', ascending=True)
        time, _ = df.time.to_numpy(), df.event.to_numpy()
        surv_durations, surv_est = KaplanMeier(df.time.to_numpy(), df.event.to_numpy())
        # this is like in pycox, see citation above
        preliminary_cuts = np.linspace(surv_est.min(), surv_est.max(), n_cuts)
        cuts_index = np.searchsorted(surv_est[::-1], preliminary_cuts)[::-1]
        final_cuts = np.unique(surv_durations[::-1][cuts_index])
        if len(final_cuts) != n_cuts:
            print(f"{len(final_cuts)} cuts are used instead of {n_cuts} since original ones are not unique.")
        # until here
        # +1 as zero time steps should not exist
        idx = np.digitize(time, final_cuts, right=False)+1

        df['time'] = idx
    return df




