# xgbsurv
XGBSurv - survival analysis with gradient boosted decision trees (GBDTs).

## Functionality

- Incorporate ties in your data with the models of Breslow and Efron.
- Directly obtain cumulative hazard and survival function as dataframes.
- Model more specific assumptions in your data with extended hazard, accelerated hazard and accelerated failure time model.
- Use convenient scikit-learn syntax to achieve quick survival predictions using gradient boosted decision trees.

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

## Data

| Dataset Name | Download URL | Load Command | Source |
|--------------|--------------|--------------|--------|
| FLCHAIN      | [Github](https://vincentarelbundock.github.io/Rdatasets/csv/survival/flchain.csv) | `load_flchain()` | [Therneau Survival in R](https://cran.r-project.org/web/packages/survival/index.html) |
| METABRIC     | [Github](https://vincentarelbundock.github.io/Rdatasets/csv/survival/flchain.csv) | `load_metabric()` | [Pycox](https://github.com/havakv/pycox) |
| RGBSG        | [Github](https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data/gbsg) | `load_rgbsg()` | [DeepSurv](https://github.com/jaredleekatzman/DeepSurv) |
| SUPPORT      | [Github](https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data/support) | `load_support()` | [DeepSurv](https://github.com/jaredleekatzman/DeepSurv) |
| TCGA         | [National Cancer Institute](https://portal.gdc.cancer.gov/) | `load_tcga(cancer_type='BLCA')` | [TCGA Research Network](https://portal.gdc.cancer.gov/)  |

## Sources & Further Reading

Chen, Tianqi, and Carlos Guestrin. 2016. “XGBoost: A Scalable Tree Boosting System.” In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–94. https://doi.org/10.1145/2939672.2939785.

Chen, Ying Qing, Nicholas P Jewell, and Jingrong Yang. 2003. “Accelerated Hazards Model: Method, Theory and Applications.” Handbook of Statistics 23: 431–41.

Kadra, Arlind, Marius Lindauer, Frank Hutter, and Josif Grabocka. 2021. “Well-Tuned Simple Nets Excel on Tabular Datasets.” Advances in Neural Information Processing Systems 34: 23928–41.

Klein, John P, and Melvin L Moeschberger. 2003. Survival Analysis: Techniques for Censored and Truncated Data. Vol. 1230. Springer.

Kvamme, Håvard, Ørnulf Borgan, and Ida Scheel. 2019. “Time-to-Event Prediction with Neural Networks and Cox Regression.” ArXiv Preprint ArXiv:1907.00825.

Mayr, Andreas, and Matthias Schmid. 2014. “Boosting the Concordance Index for Survival Data–a Unified Framework to Derive and Evaluate Biomarker Combinations.” PloS One 9 (1): e84483.

Tseng, Yi-Kuan, and Ken-Ning Shu. 2011. “Efficient Estimation for a Semiparametric Extended Hazards Model.” Communications in Statistics—Simulation and Computation® 40 (2): 258–73.

Zhong, Qixian, Jonas W Mueller, and Jane-Ling Wang. 2021. “Deep Extended Hazard Models for Survival Analysis.” Advances in Neural Information Processing Systems 34: 15111–24.

## Citation

If you would like to cite this work:

jatlantic. 2023. "XGBSurv: Gradient Boosted Decision Trees for Survival Analysis."







