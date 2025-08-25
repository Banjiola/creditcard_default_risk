import time 
import joblib
from scipy.stats import randint, loguniform 
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from imblearn.pipeline import Pipeline as imb_pipeline
from imblearn.over_sampling import SMOTE
import pandas as pd
from model_selection.model_training import load_model
from pathlib import Path

random_state= 70 
CV = StratifiedKFold(n_splits= 5,
                     shuffle= True, #splits
                     random_state= random_state)

# Define Parameter Distribution
tree_param_dist = {
    'tree__criterion': ['gini', 'entropy'],
    'tree__max_depth': [3, 5, 7, 10, 15],
    'tree__min_samples_split': [20, 50, 100, 200], 
    'tree__min_samples_leaf': [10, 20, 50, 100],   
    'tree__max_features': ['sqrt', 'log2', 0.5],
    'tree__max_leaf_nodes': [50, 100, 200, 500] 
}

lr_param_dist = {
    'lr__C': loguniform(0.01, 100),           
    'lr__class_weight': [None, 'balanced'],  
    'lr__max_iter': randint(100, 2000),       
    'lr__solver': ['liblinear', 'lbfgs'],   
    'lr__penalty': ['l1', 'l2']             
}

svm_param_dist = {
    'svm__C': [0.1, 1, 10],  
    'svm__kernel': ['rbf', 'sigmoid'],  
    'svm__gamma': ['scale', 0.1],  
    'svm__class_weight': [None, 'balanced']  
}
knn_param_dist= {
    'knn__algorithm': ['auto', 'ball_tree', 'brute'], 
    'knn__metric': ['minkowski', 'euclidean'],     
    'knn__n_neighbors': [5, 11, 25,77, 181, 201],
    'knn__weights': ['uniform', 'distance']
    }

xgb_param_dist = {
    'xgb__n_estimators': randint(50, 200), 
    'xgb__max_depth': randint(3, 10), 
    'xgb__min_child_weight': randint(1, 10),
    'xgb__learning_rate': loguniform(0.01, 0.3),
    'xgb__reg_alpha': loguniform(0.01, 10),        
    'xgb__reg_lambda': loguniform(0.01, 10)   # for reproducibility the random_state will be passed into the randomisedsearchCV
    }


def tune_params(
        clf, param_dist,
        X_train, y_train,
        random_state=random_state,CV=CV,
        n_iter=30,
        scoring={
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'pr_auc': 'average_precision'
            }):
   
    """
    Perform hyperparameter tuning using RandomizedSearchCV with multiple scoring metrics.

    This function uses randomized search to find optimal hyperparameters for a classifier
    by evaluating multiple scoring metrics and selecting the best parameters based on F1 score.

    Parameters
    ----------
    clf : estimator/pipeline object
        The classifier to tune (e.g., RandomForestClassifier, SVC, etc.)
    param_dist : dict
        Dictionary with parameter names as keys and distributions
        or lists of parameter settings to try as values
    X_train : pd.DataFrame, array-like
        Feature Matrix.
    y_train : pd.Series, array-like
        Target Column.
    n_iter : int, optional
        Number of parameter settings sampled from param_dist.
        Defaults to 30.
    random_state : int
        Random state for reproducible results
    scoring : dict, optional
        Dictionary of scoring metrics to evaluate during
        cross-validation. Defaults to precision, recall, f1, and pr_auc.
       
    Returns
    -------
    tuple
       A tuple containing:
       - best_clf: The best estimator found by the search, fitted on training data
       - training time (flaot): Time taken to complete training process (in seconds)
       - duration (float): Time taken to complete tuning process (in seconds)
       - random_search object.
    """
    
    random_search = RandomizedSearchCV(estimator = clf,
                                       param_distributions = param_dist,
                                       n_iter = n_iter,
                                       cv = CV, #stratified k fold has been defined earlier
                                       scoring = scoring,
                                       refit='f1', # This is the overall parameter we are trying to optimise
                                       n_jobs= -1, # I opted to use all of my CPU's 8 cores as the SVM experiment lasted 6 hrs+
                                       verbose= 2, # This shows some progress of the cv search 
                                       random_state = random_state)
    
    # Tuning Time
    start_tune = time.time()
    random_search.fit(X_train,y_train)
    stop_tune = time.time()
    tuning_time = stop_tune - start_tune

    
    # Best estimator
    best_clf = random_search.best_estimator_

    # training time
    start = time.time()
    best_clf.fit(X_train, y_train)
    training_time = time.time() - start
    
    return best_clf, tuning_time, training_time, random_search

def save_random_search_object(object, object_name):
    """Saves a random object to appropriate
    directory"""
    
    
    folder = Path("../models/random_search_objects")
    folder.mkdir(parents=True, exist_ok=True) 

    # Full path
    object_path = folder / f"{object_name}.joblib"
        
    # Save the model
    joblib.dump(object, object_path)
    print(f"{object_name} saved at {object_path}") # remove later
    return f"{object_name} saved at {object_path}"

def load_random_search_object(object_name):
    """Load a saved random object from the models directory"""
    object_path = Path("../models/random_search_objects") / f"{object_name}.joblib"
    
    # Load the model
    object = joblib.load(object_path)
    print(f"{object_name} loaded from {object_path}") # remove later
    return object    

if __name__ =='__main__':
    print('here1')
    scaled_X_train_with_engineering = pd.read_csv("../../datasets/scaled_X_train_engineering.csv")
    print('here2')
    X_test= pd.read_csv("../data/X_test.csv")
    y_train= pd.read_csv("../../datasets/y_train.csv").squeeze()
    y_test = pd.read_csv("../../datasets/y_test.csv").squeeze()
   
    tree = load_model("dec_tree_with_feature_engineering")
    log_reg = load_model("log_reg_with_feature_engineering")
    svm = load_model("svm_with_feature_engineering")
    knn = load_model("knn_with_feature_engineering")
    xgb = load_model("xgb_feature_engineering")
    
    # here i will have to import the baseline classifiers
    
    tree_pipeline = imb_pipeline([
        ('smote', SMOTE(random_state=random_state)),
        ('tree', tree)
        ])

    log_reg_pipeline = imb_pipeline([
    ('smote', SMOTE(random_state=random_state)),
    ('lr', log_reg)
        ])

    svm_pipeline = imb_pipeline([
    ('smote', SMOTE(random_state=random_state)),
    ('svm', svm)
        ])

    knn_pipeline = imb_pipeline([
    ('smote', SMOTE(random_state=random_state)),
    ('knn', knn)
        ])

    xgb_pipeline = imb_pipeline([
    ('smote', SMOTE(random_state=random_state)),
    ('xgb', xgb)
        ])
    
    # now we implement the random search
    tree_random_search = tune_params(
        clf= tree_pipeline, 
        X_train=scaled_X_train_with_engineering, 
        y_train=y_train,
        param_dist= tree_param_dist
        )
    print("Tuning of Decision Tree Complete".center(100))
    print('='*100)
'''
    # Run randomised search
    log_reg_random_search = tune_params(
        clf= log_reg_pipeline,
        X_train=scaled_X_train_with_engineering, 
        y_train=y_train,
        param_dist= lr_param_dist)
    print("Tuning of Logistic Regression Complete".center(100))
    print('='*100)

    svm_random_search = tune_params(
        clf= svm_pipeline,
        X_train=scaled_X_train_with_engineering, 
        y_train=y_train,
        param_dist= svm_param_dist,
        n_iter= 12)
    
    print("Tuning of SVM Complete".center(100))
    print('='*100)

    knn_random_search = tune_params(
        clf= knn_pipeline,
        X_train=scaled_X_train_with_engineering, 
        y_train=y_train,
        param_dist= knn_param_dist)
    
    print("Tuning of KNN Complete".center(100))
    print('='*100)    
    
    xgb_random_search = tune_params(
        clf= xgb_pipeline,
        X_train=scaled_X_train_with_engineering, 
        y_train=y_train,
        param_dist= xgb_param_dist)
    print("Tuning of XGB Complete".center(100))
    print('='*100)'''
save_random_search_object(tree_random_search, 'decision tree')