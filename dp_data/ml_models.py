from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier

MODELS = {
   'DecisionTree': DecisionTreeClassifier(), 
   'KNN': KNeighborsClassifier(),
   'LogisticRegression': LogisticRegression(solver='liblinear'),
   'LinearSVC': LinearSVC(),
   'RandomForest': RandomForestClassifier(),
   'AdaBoost': AdaBoostClassifier(),
   'GradientBoosting': GradientBoostingClassifier(),
   'XGBoost': XGBClassifier(),
   # 'LightGBM': LGBMClassifier(),
}
MODEL_PARAMS = {
   'DecisionTree': {},
   'KNN': {},
   'LogisticRegression': {},
   'SVM': {},
   'RandomForest': {},
   'AdaBoost': {},
   'GradientBoosting': {},
   'XGBoost': {},
   # 'LightGBM': {},
}