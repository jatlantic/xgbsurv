import pandas as pd
import os
from sklearn.datasets._base import _convert_data_dataframe
from xgbsurv.datasets.utils import transform
#/Users/JUSC/Documents/xgbsurv/xgbsurv/datasets/data/METABRIC_adapted.csv

def load_metabric(*, path="datasets/data/", return_X_y=True, as_frame=False):
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

    # if as_frame:
    #     frame, data, target = _convert_data_dataframe(
    #         "load_iris", data, target, feature_names, target_columns
    #     )
    
    if os.path.exists(path+data_file_name):
        df = pd.read_csv(path+data_file_name)
    else:
        current_directory = os.getcwd()
        raise FileNotFoundError(str(path+data_file_name)+" is not a valid path. Start from "+str(current_directory)+"!")

    # TODO: sort metabric and activate check
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
    if return_X_y:
        return data, target
    else:
        raise NotImplementedError('Bunch object is not implemented.')

        # return Bunch(
        #     data=data,
        #     target=target,
        #     frame=frame,
        #     target_names=target_names,
        #     DESCR=fdescr,
        #     feature_names=feature_names,
        #     filename=data_file_name,
        #     data_module=DATA_MODULE,
        # )

def check_if_time_sorted(dataframe):
    check_sorted = dataframe["time"].is_monotonic_increasing
    return check_sorted



#TODO: Implement checksum with hashlib, build check function for time=0 and sorting!

