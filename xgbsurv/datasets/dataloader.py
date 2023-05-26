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
    # datatypes
    datatypes = {
    'MKI67': np.float32,
    'EGFR': np.float32, 
    'PGR': np.float32,
    'ERBB2': np.float32, 
    'hormone_treatment': np.uint8, 
    'radiotherapy': np.uint8, 
    'chemotherapy': np.uint8, 
    'ER_positive': np.uint8, 
    'age': np.float32,
    #'time':np.float32,
    #'event': np.float32
    }

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

    if as_frame:
        frame, data, target = _convert_data_dataframe(
            "load_metabric", data, target, feature_names, target_columns
        )
        data = data.astype(datatypes)
        #return data, target
    
    # make sure data type is correct
    target = target.astype(np.float32)


    if return_X_y:
        return data, target

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

    datatypes = {
    'age': np.float32,
    'sex': np.uint8, 
    'sample_yr': np.float32, 
    'kappa': np.float32, 
    'lambda': np.float32, 
    'flc_grp': np.float32, 
    'creatinine': np.float32, 
    'mgus': np.uint8}

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
        data = data.astype(datatypes)
    target = target.astype(np.float32)

    
    if return_X_y:
        return data, target

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

    datatypes = {
        'age': np.float32,
        'sex': np.uint8, 
        'race': np.object_,
        'n_comorbidities': np.float32,
        'diabetes': np.uint8, 
        'dementia': np.uint8,
        'cancer': np.object_,
        'blood_pressure': np.float32,
        'heart_rate': np.float32,
        'respiration_rate': np.float32,
        'temperature': np.float32, 
        'white_blood_cell': np.float32, 
        'serum_sodium': np.float32, 
        'serum_creatinine': np.float32}

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
        data = data.astype(datatypes)
        # categorical vars
        data['cancer'] = data['cancer'].astype('category')
        data['race'] = data['race'].astype('category')
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
    datatypes = {
    'horm_treatment': np.uint8, 
    'grade': np.object_,
    'menopause': np.uint8, 
    'age': np.float32, 
    'n_positive_nodes': np.float32, 
    'progesterone': np.float32, 
    'estrogene': np.float32}

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
        data = data.astype(datatypes)
    #data['horm_treatment'] = data['horm_treatment'].astype('category')
    #data['menopause'] = data['menopause'].astype('category')
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