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

We also provide code for training classifiers on such data. For example,
````
DATASET=adult
MODELS='DecisionTree KNN LogisticRegression LinearSVC RandomForest GradientBoosting XGBoost'
python run/ml_eval.py --dataset $DATASET --train_test_split_dir original --models $MODELS
````