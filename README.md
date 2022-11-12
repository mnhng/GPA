# GPA - Generalized Prevalence Adjustment

Healthcare data often come from multiple sites in which the correlations between the target (Y) and the confounding variables (Z) can vary widely.
If deep learning models exploit these unstable correlations, they might fail catastrophically in unseen sites.
GPA is a flexible method that adjusts model predictions to the shifting correlations between prediction target and confounders to safely exploit unstable features.

GPA learns a stable adaptive predictor by modeling the data-generation process using two separate estimators: one for the stable mechanism and the other for the shifting P(Y|Z) distribution.
GPA can infer the interaction between the target and the confounders at new sites using only unlabeled samples from those sites.
Furthermore, GPA can better predict for samples without Z at test time even for high-dimensional Z.

Link to paper: [Adapting to Shifting Correlations with Unlabeled Data Calibration](https://arxiv.org/abs/2409.05996)

## How to Setup

1. Install [*miniconda*](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)
2. Setup the conda environment
```
    conda env create -n gpa
    conda activate gpa
    conda install torchvision pytorch-cuda -c pytorch -c nvidia
    conda install numpy scipy scikit-learn pandas
```

## How to Use

1. Activate the conda environment
2. Execute the shell scripts in the `examples` folder. 
3. For ISIC experiment, see the instruction in the `data/ISIC` folder.
4. For chest X-Ray experiment, download the datasets from [PhysioNet](https://physionet.org/).

## Cite

```
@inproceedings{nguyen2024adapting,
    title={{Adapting to Shifting Correlations with Unlabeled Data Calibration}},
    author={Nguyen, Minh and Wang, Alan Q. and Kim, Heejong and Sabuncu, Mert R.},
    booktitle={Proceedings of ECCV},
    year={2024}
}
@inproceedings{nguyen2024robust,
    title={{Robust Learning via Conditional Prevalence Adjustment}},
    author={Nguyen, Minh and Wang, Alan Q., and Kim, Heejong and Sabuncu, Mert R.},
    booktitle={Proceedings of WACV},
    year={2024}
}
```
