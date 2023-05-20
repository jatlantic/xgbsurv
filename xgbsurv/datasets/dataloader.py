import pandas as pd
import numpy as np
import os
from sklearn.datasets._base import _convert_data_dataframe
from sklearn.utils import Bunch
from xgbsurv.datasets.utils import transform
import warnings
warnings.filterwarnings("ignore")
#/Users/JUSC/Documents/xgbsurv/xgbsurv/datasets/data/METABRIC_adapted.csv

def check_if_time_sorted(dataframe):
    check_sorted = dataframe["time"].is_monotonic_increasing
    return check_sorted


def load_metabric(*, path="datasets/data/", return_X_y=False, as_frame=False):
    """add sklearn style documentation here
    """
    
    data_file_name = "METABRIC_adapted.csv"
    # potentially use sklearn load_csv_data
    #data, target, target_names, fdescr = load_csv_data(
    #    data_file_name=data_file_name, descr_file_name="iris.rst"
    #)


    feature_names = [
        "MKI67",
    	"EGFR",
        "PGR",
        "ERBB2",
        "hormone_treatment",
        "radiotherapy",
        "chemotherapy",
        "ER_positive",
        "age"
    ]

    frame = None
    target_columns = [
        "target"
    ]

    if os.path.exists(path+data_file_name):
        df = pd.read_csv(path+data_file_name)
    else:
        current_directory = os.getcwd()
        raise FileNotFoundError(str(path+data_file_name)+" is not a valid path. Start from "+str(current_directory)+"!")

    #feature_names = df.columns.tolist()
    # sort metabric and activate check
    if check_if_time_sorted(df):
        pass
    else:
        raise ValueError("Dataset is not sorted by time ascending order.")
    
    # transform data
    target = transform(df.time.to_numpy(),df.event.to_numpy())
    data = df.iloc[:,:-2].to_numpy()

    # make sure data type is correct
    data, target = data.astype(np.float32), target.astype(np.float32)

    if as_frame:
        frame, data, target = _convert_data_dataframe(
            "load_metabric", data, target, feature_names, target_columns
        )
        #return data, target
    
    if return_X_y:
        return data, target
    #else:
        #raise NotImplementedError('Bunch object is not implemented.')

    return Bunch(
            data=data,
            target=target,
            frame=frame,
            feature_names=feature_names,
            filename=data_file_name,
        )

#TODO: Implement checksum with hashlib, build check function for time=0 and sorting!

def load_flchain(*, path="datasets/data/", return_X_y=False, as_frame=False):
    """add sklearn style documentation here
    """
    # TODO: preprocessing for orig data
    data_file_name = "FLCHAIN_adapted.csv"
    # potentially use sklearn load_csv_data
    #data, target, target_names, fdescr = load_csv_data(
    #    data_file_name=data_file_name, descr_file_name="iris.rst"
    #)


    feature_names = [
        "age",
        "sex",
        "sample_yr",
        "kappa",
        "lambda",
        "flc_grp",
        "creatinine",
        "mgus"
    ]

    frame = None
    target_columns = [
        "target"
    ]

    if os.path.exists(path+data_file_name):
        df = pd.read_csv(path+data_file_name)
    else:
        current_directory = os.getcwd()
        raise FileNotFoundError(str(path+data_file_name)+" is not a valid path. Start from "+str(current_directory)+"!")

    # set datatypes

    if check_if_time_sorted(df):
        pass
    else:
        #df.sort_values(by='time', inplace=True)
        raise ValueError("Dataset is not sorted by time ascending order.")
    
    # transform data
    target = transform(df.time.to_numpy(),df.event.to_numpy())
    data = df.iloc[:,:-2].to_numpy()

    if as_frame:
        frame, data, target = _convert_data_dataframe(
            "load_flchain", data, target, feature_names, target_columns
        )
        data = data.astype(np.float32)
        # change to categorical
        data['sex'] = data['sex'].astype('category')
        data['mgus'] = data['mgus'].astype('category')
        target = target.astype(np.float32)
        #print(target.dtypes)
        #print(data.dtypes)
        #return data, target
    
    if return_X_y:
        return data, target
    #else:
        #raise NotImplementedError('Bunch object is not implemented.')

    return Bunch(
            data=data,
            target=target,
            frame=frame,
            feature_names=feature_names,
            filename=data_file_name,
        )
    

def load_support(*, path="datasets/data/", return_X_y=False, as_frame=False):
    """add sklearn style documentation here
    """
    # TODO: preprocessing for orig data
    data_file_name = "SUPPORT_adapted.csv"
    # potentially use sklearn load_csv_data
    #data, target, target_names, fdescr = load_csv_data(
    #    data_file_name=data_file_name, descr_file_name="iris.rst"
    #)


    feature_names = [
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
        "serum_creatinine"
    ]

    frame = None
    target_columns = [
        "target"
    ]

    if os.path.exists(path+data_file_name):
        df = pd.read_csv(path+data_file_name)
    else:
        current_directory = os.getcwd()
        raise FileNotFoundError(str(path+data_file_name)+" is not a valid path. Start from "+str(current_directory)+"!")

    #feature_names = df.columns.tolist()
    # sort metabric and activate check
    if check_if_time_sorted(df):
        pass
    else:
        #df.sort_values(by='time', inplace=True)
        raise ValueError("Dataset is not sorted by time ascending order.")
    
    # transform data
    target = transform(df.time.to_numpy(),df.event.to_numpy())
    data = df.iloc[:,:-2].to_numpy()

    if as_frame:
        frame, data, target = _convert_data_dataframe(
            "load_support", data, target, feature_names, target_columns
        )
        data = data.astype(np.float32)
        # categorical vars
        data['sex'] = data['sex'].astype('category')
        data['diabetes'] = data['diabetes'].astype('category')
        data['dementia'] = data['dementia'].astype('category')
        data['cancer'] = data['cancer'].astype('category')
        target = target.astype(np.float32)
    
    if return_X_y:
        return data, target
    #else:
        #raise NotImplementedError('Bunch object is not implemented.')

    return Bunch(
            data=data,
            target=target,
            frame=frame,
            feature_names=feature_names,
            filename=data_file_name,
        )
    

def load_rgbsg(*, path="datasets/data/", return_X_y=False, as_frame=False):
    """add sklearn style documentation here
    """
    # TODO: preprocessing for orig data
    data_file_name = "RGBSG_adapted.csv"
    # potentially use sklearn load_csv_data
    #data, target, target_names, fdescr = load_csv_data(
    #    data_file_name=data_file_name, descr_file_name="iris.rst"
    #)


    feature_names = [
        "horm_treatment",
        "grade",
        "menopause",
        "age",
        "n_positive_nodes",
        "progesterone",
        "estrogene"
    ]

    frame = None
    target_columns = [
        "target"
    ]

    if os.path.exists(path+data_file_name):
        df = pd.read_csv(path+data_file_name)
    else:
        current_directory = os.getcwd()
        raise FileNotFoundError(str(path+data_file_name)+" is not a valid path. Start from "+str(current_directory)+"!")

    #feature_names = df.columns.tolist()
    if check_if_time_sorted(df):
        pass
    else:
        #df.sort_values(by='time', inplace=True)
        raise ValueError("Dataset is not sorted by time ascending order.")
    
    # transform data
    target = transform(df.time.to_numpy(),df.event.to_numpy())
    data = df.iloc[:,:-2].to_numpy()

    if as_frame:
        frame, data, target = _convert_data_dataframe(
            "load_rgbsg", data, target, feature_names, target_columns
        )
        #return data, target
        data = data.astype(np.float32)
        data['horm_treatment'] = data['horm_treatment'].astype('category')
        data['menopause'] = data['menopause'].astype('category')
        data['grade'] = data['grade'].astype('category')
        target = target.astype(np.float32)
    
    if return_X_y:
        return data, target
    #else:
        #raise NotImplementedError('Bunch object is not implemented.')

    return Bunch(
            data=data,
            target=target,
            frame=frame,
            feature_names=feature_names,
            filename=data_file_name,
        )
    

def load_tcga(*, cancer_type='BLCA' ,path="datasets/data/", return_X_y=False, as_frame=False):
    """add sklearn style documentation here
    """
    # TODO: preprocessing for orig data
    data_file_name = cancer_type+"_adapted.csv"
    # potentially use sklearn load_csv_data
    #data, target, target_names, fdescr = load_csv_data(
    #    data_file_name=data_file_name, descr_file_name="iris.rst"
    #)

    frame = None
    target_columns = [
        "target"
    ]

    if os.path.exists(path+data_file_name):
        df = pd.read_csv(path+data_file_name)
    else:
        current_directory = os.getcwd()
        raise FileNotFoundError(str(path+data_file_name)+" is not a valid path. Start from "+str(current_directory)+"!")

    
    #feature_names = df.columns.tolist()
    if check_if_time_sorted(df):
        pass
    else:
        #df.sort_values(by='time', inplace=True)
        raise ValueError("Dataset is not sorted by time ascending order.")
    
    # transform data
    df = df[df['time']>0]
    target = transform(df.time.to_numpy(), df.event.to_numpy())
    target = target.astype(np.float32)
    data = df.iloc[:, 2:].to_numpy()
    data = data.astype(np.float32)
    feature_names = df.iloc[:,2:].columns

    # change name potentially to cancer type
    if as_frame:
        frame, data, target = _convert_data_dataframe(
            "load_tcga", data, target, feature_names, target_columns
        )
    #    return data, target
    
    if return_X_y:
        return data, target
    #else:
        #raise NotImplementedError('Bunch object is not implemented.')

    return Bunch(
            data=data,
            target=target,
            frame=frame,
            feature_names=feature_names,
            filename=data_file_name,
        )