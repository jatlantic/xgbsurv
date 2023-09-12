from pathlib import Path, PosixPath
import os
import numpy as np
import pandas as pd

#TEST_DIR = Path(__file__).parent
# so it can be used in interactive shell
TEST_DIR = os.path.dirname(os.path.abspath('__file__'))

def get_1d_array(case="default", dims=1):
    file_path = PosixPath(TEST_DIR) / "xgbsurv" / "tests" /"test_data" / f"survival_simulation_25_{case}.csv"
    print('file path',file_path)
    df = pd.read_csv(file_path)

    linear_predictor = df.preds.to_numpy(dtype=np.float32)
    time = df.time.to_numpy(dtype=np.float32)
    event = df.event.to_numpy(dtype=np.float32)
    return linear_predictor, time, event

# change dir
def get_2d_array(case="default"):
    file_path = PosixPath(TEST_DIR)/ "xgbsurv" / "tests" /"test_data" / f"survival_simulation_25_{case}.csv"
    df = pd.read_csv(file_path)

    pred_1d = df.preds.to_numpy(dtype=np.float32)[:, None]
    linear_predictor = np.hstack((pred_1d, pred_1d))
    time = df.time.to_numpy(dtype=np.float32)
    event = df.event.to_numpy(dtype=np.float32)
    return linear_predictor, time, event