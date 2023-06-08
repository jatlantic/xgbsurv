# xgbsurv
Sklearn survival analysis with gradient boosted decision trees (GBDTs).

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

## Implemented Models & Loss Functions

| Model                     | Loss Function | GBDT | DL   |
|---------------------------|---------------|------|------|
| Breslow                   | $L_{\mathrm{breslow}}=\prod_i\left(\frac{\exp \left[g\left(\mathbf{x}_i\right)\right]}{\sum_{j \in \mathcal{R}_i} \exp \left[g\left(\mathbf{x}_j\right)\right]}\right)^{D_i}$        | ✔️   | ✔️   |
| Efron                     | Loss 2        | ✔️   | ✔️   |
| Cboost                    | Loss 3        | ✔️   | ✔️   |
| Extended Hazards          | Loss 3        | ✔️   | ✔️   |
| Accelerated Hazards       | Loss 3        | ✔️   | ✔️   |
| Accelerated Failure Time  | Loss 3        | ✔️   | ✔️   |


## Experiments

Breslow
