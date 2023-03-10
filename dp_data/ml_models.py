from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

MODELS = {
   'DecisionTree': DecisionTreeClassifier(class_weight='balanced'), 
   'KNN': KNeighborsClassifier(),
   'LogisticRegression': LogisticRegression(class_weight='balanced', solver='sag'),
   'LinearSVC': LinearSVC(class_weight='balanced'),
   'RandomForest': RandomForestClassifier(class_weight='balanced'),
   'AdaBoost': AdaBoostClassifier(),
   'GradientBoosting': GradientBoostingClassifier(),
   'XGBoost': XGBClassifier(),
   'LightGBM': LGBMClassifier(),
}
MODEL_PARAMS = {
   'DecisionTree': {},
   'KNN': {},
   'LogisticRegression': {},
   'SVM': {},
   'RandomForest': {},
   'AdaBoost': {},
   'GradientBoosting': {},
   'AdaBoost': {},
   'XGBoost': {},
   'LightGBM': {},
}