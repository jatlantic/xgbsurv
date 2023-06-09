# xgbsurv
Sklearn survival analysis with gradient boosted decision trees (GBDTs).


## Implemented Models & Experiments

| Models                    | GBDT | DL   | Example Notebook   |
|---------------------------|------|------|--------------------|
| Breslow                   | ✔️    | ✔️    |[Breslow]([../examples/xgbsurv_breslow.ipynb](https://github.com/jatlantic/xgbsurv/blob/main/examples/xgbsurv_breslow.ipynb))|
| Efron                     | ✔️    | ✔️    |[Efron](../xgbsurv/examples/xgbsurv_efron.ipynb)|
| Cboost                    | ✔️    | ✔️    |[Cboost](../xgbsurv/examples/xgbsurv_cboost.ipynb)|
| Extended Hazards          | ✔️    | ✔️    |[EH](../xgbsurv/examples/xgbsurv_eh.ipynb)|
| Accelerated Hazards       | ✔️    | ✔️    |[AH](../xgbsurv/examples/xgbsurv_ah.ipynb)|
| Accelerated Failure Time  | ✔️    | ✔️    |[AFT](../xgbsurv/examples/xgbsurv_aft.ipynb)|



## Installation

Create a conda environment:

```console
conda env create -f conda.yml
```

Activate the environment:

```console
conda activate xgbsurv
```

Install:

```console
pip install .
```

### development

Install in editable mode for development:

```sh
pip install --user -e .
```





