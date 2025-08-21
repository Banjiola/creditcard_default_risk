from evaluation_utils import get_train_metrics
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.pipeline import Pipeline 
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold


random_state = 70
cv = StratifiedKFold(n_splits= 5,
                     shuffle= True, #splits
                     random_state= random_state)


tree = joblib.load("../models/dec_tree_with_feature_engineering.joblib")
log_reg = joblib.load("../models/log_reg_with_feature_engineering.joblib")
svm = joblib.load("../models/svm_with_feature_engineering.joblib")
knn = joblib.load("../models/knn_with_feature_engineering.joblib")
xgb = joblib.load("../models/xgb_feature_engineering.joblib")
get_train_metrics(tree, "Decision Tree",X_train_with_featured, y_train )
get_train_metrics(log_reg, "log reg",X_train_with_featured, y_train )
get_train_metrics(svm, "SVM",X_train_with_featured, y_train )
get_train_metrics(knn, "KNN",X_train_with_featured, y_train )
get_train_metrics(xgb, "XGB",X_train_with_featured, y_train )
