import pandas as pd
from evaluation_utils import get_cv_metrics, get_train_metrics
from sklearn.model_selection import StratifiedKFold
from model_selection.model_training import load_model

RANDOM_STATE = 70
CV = StratifiedKFold(n_splits= 5,
                     shuffle= True, #splits
                     random_state= RANDOM_STATE)

if __name__=="__main__":

    # Load data
    scaled_X_train_with_engineering = pd.read_csv("datasets/scaled_X_train_engineering.csv")
    X_test= pd.read_csv("datasets/X_test.csv")
    y_train= pd.read_csv("datasets/y_train.csv").squeeze()
    y_test = pd.read_csv("datasets/y_test.csv").squeeze()
    
    # load models trained on scaled_X_train_engineering
    tree = load_model("dec_tree_with_feature_engineering")
    log_reg = load_model("log_reg_with_feature_engineering")
    svm = load_model("svm_with_feature_engineering")
    knn = load_model("knn_with_feature_engineering")
    xgb = load_model("xgb_feature_engineering")

    print("Evaluation Metrics of Untuned Models on Training Data".upper().center(75,'='))
    get_train_metrics(tree, "Decision Tree",scaled_X_train_with_engineering, y_train)
    get_train_metrics(log_reg, "Logistic Regression",scaled_X_train_with_engineering, y_train )
    get_train_metrics(svm, "Support Vector Machine",scaled_X_train_with_engineering, y_train )
    get_train_metrics(knn, "K-Nearest Neighbour",scaled_X_train_with_engineering, y_train )
    get_train_metrics(xgb, "Xtreme Gradient Boost",scaled_X_train_with_engineering, y_train )

    print("Evaluation Metrics of Untuned Models on Cross Validation Data".upper().center(75,'='))
    get_cv_metrics(tree, "Decision Tree",scaled_X_train_with_engineering, y_train,cv=CV)
    get_cv_metrics(log_reg, "Logistic Regression",scaled_X_train_with_engineering, y_train,cv=CV )
    get_cv_metrics(svm, "Support Vector Machine",scaled_X_train_with_engineering, y_train,cv=CV )
    get_cv_metrics(knn, "K-Nearest Neighbour",scaled_X_train_with_engineering, y_train,cv=CV )
    get_cv_metrics(xgb, "Xtreme Gradient Boost",scaled_X_train_with_engineering, y_train ,cv=CV)