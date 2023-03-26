import pandas as pd

# original data is taken from pycox
# https://github.com/havakv/pycox/tree/master/pycox/datasets

def metabric_preprocess(path="add your path here"):
    filename="original_data/METABRIC_pycox_full.csv"
    df = pd.read_csv(path+filename)
    # name columns
    df.columns = ['MKI67', 'EGFR', 'PGR', 'ERBB2', 'hormone_treatment', 'radiotherapy', 'chemotherapy', 'ER_positive', 'age', 'time', 'event']
    # remove zero time observations
    df = df[df.time!=0]
    # sort data
    df.sort_values(by='time', ascending=True, inplace=True)
    # save data
    df.to_csv(path+"data/METABRIC_adapted.csv", index=False)
    return

#metabric_preprocess()

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




