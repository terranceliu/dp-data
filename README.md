# Setup

Our codebase currently supports Python **3.9**. We recommend that you create a separate virtual or Conda environment.

For example,
````
conda create -n dp-data python=3.9
````

Install the source files (via setuptools)
````
conda activate dp-data
pip install --upgrade pip
pip install -e .
````

# Execution

Scripts for preprocessing individual datasets can be found in `scripts`. The following script will preprocess all available datasets.
````
./preprocess_all.sh
````
Note that the `credit` dataset must be downloaded manually from [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and moved to `datasets/raw/credit.csv`. Please do this first, and then run the following command separately.
````
./scripts/credit.sh
````

## Running simple classification baselines
We provide a script for training classifiers our preprocessed datasets. Our scripts by default creates 80-20 train-test splits. An example of how to use this code can be found below:
````
DATASET=adult
MODELS='DecisionTree KNN LogisticRegression LinearSVC RandomForest GradientBoosting XGBoost'
python run/ml_eval.py --dataset $DATASET --train_test_split_dir original --models $MODELS
````

## Census Privacy Protected Microdata Files (PPMF)

Our scripts generate a large set of datasets from the [PPMF](https://www.census.gov/programs-surveys/decennial-census/decade/2020/planning-management/process/disclosure-avoidance/2020-das-development.html) raw source data, and the overall preprocessing pipeline may take several hours. We therefore do not include them in `preprocess_all.sh`. To obtain them, you can instead run the following scripts separately.
````
./scripts/ppmf/tracts.sh
./scripts/ppmf/blocks.sh
````

Please also note that the PPMF datasets are constructed with 50-50 train-test splits. The purpose of such splits is to reproduce baseline results for the experiments found in [this work](https://arxiv.org/abs/2211.03128). There is no natural classification problem for this data, but you are free to construct your own by specifying a `target` attribute when running `run/ml_eval.py`.