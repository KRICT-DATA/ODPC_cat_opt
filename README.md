This repository contains source code and dataset for the following paper: "Closed-loop optimization of catalyst for oxidative propane dehydrogenation with CO2 (ODPC) using artificial intelligence" [link](https://doi.org/10.1016/j.jcou.2023.102620)

## 0. Prerequisites

* Python 3.7 or later
* [NumPy](https://numpy.org/install/) 
* [Pandas](https://pandas.pydata.org/docs/getting_started/index.html#getting-started)
* [Scikit-learn](https://scikit-learn.org/stable/install.html)
* [joblib](https://joblib.readthedocs.io/en/stable/installing.html)
* [Matplotlib](https://matplotlib.org/stable/) (for plotting)


## 1. Data

1.1 ODPC experiment initial 200 dataset: "ODPC Initial Data.xlsx"

1.2 ODPC experiment final 200+27 dataset: "ODPC Final Data.xlsx"



## 2. Source code

2.1 Leave-one-out cross-validation (LOOCV) of gradient-boosted decision trees (GBDT) model and final training: type "python LOOCV_and_model_training.py"

2.2 Execute artificial bee colony (ABC) algorithm for propylene yield: type "python ABC_exec_Yield.py"

2.3 Execute artificial bee colony (ABC) algorithm for the score: type "python ABC_exec_Score.py"


After executing ABC algorithm for the yield and score, you will get "sol_GBR_yield.csv" and "sol_GBR_score.csv" files. These files contain catalyst composition and its predicted yield and score. 
