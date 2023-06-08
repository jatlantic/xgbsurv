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
| Breslow                   | L<sub>breslow</sub> = ∏<sub>i</sub>((e<sup>g(x<sub>i</sub>)</sup>)/(∑<sub>j ∈ R<sub>i</sub></sub>e<sup>g(x<sub>j</sub>)</sup>))<sup>D<sub>i</sub></sup>         | ✔️   | ✔️   |
| Efron                     | Loss 2        | ✔️   | ✔️   |
| Cboost                    | Loss 3        | ✔️   | ✔️   |
| Extended Hazards          | Loss 3        | ✔️   | ✔️   |
| Accelerated Hazards       | Loss 3        | ✔️   | ✔️   |
| Accelerated Failure Time  | Loss 3        | ✔️   | ✔️   |


## Experiments

Breslow
