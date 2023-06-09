# xgbsurv
Sklearn survival analysis with gradient boosted decision trees (GBDTs).


## Implemented Models & Experiments

| Models                    | GBDT | DL   | Example Notebook   |
|---------------------------|------|------|--------------------|
| Breslow                   | ✔️    | ✔️    |[Breslow](https://github.com/jatlantic/xgbsurv/blob/main/examples/xgbsurv_breslow.ipynb)|
| Efron                     | ✔️    | ✔️    |[Efron](https://github.com/jatlantic/xgbsurv/blob/main/examples/xgbsurv_efron.ipynb)|
| Cboost                    | ✔️    | ✔️    |[Cboost](https://github.com/jatlantic/xgbsurv/blob/main/examples/xgbsurv_cboost.ipynb)|
| Extended Hazards          | ✔️    | ✔️    |[EH](https://github.com/jatlantic/xgbsurv/blob/main/examples/xgbsurv_eh.ipynb)|
| Accelerated Hazards       | ✔️    | ✔️    |[AH](https://github.com/jatlantic/xgbsurv/blob/main/examples/xgbsurv_ah.ipynb)|
| Accelerated Failure Time  | ✔️    | ✔️    |[AFT](https://github.com/jatlantic/xgbsurv/blob/main/examples/xgbsurv_aft.ipynb)|


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





