from scipy.stats import randint, loguniform 
import time 
random_state= 70 
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict


cv = StratifiedKFold(n_splits= 5,
                     shuffle= True, #splits
                     random_state= random_state)

def tune_params(
        clf, param_dist,
        X_train, y_train,
        random_state=random_state,cv=cv,
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
                                       cv = cv, #stratified k fold has been defined earlier
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

if __name__ =='__main__':
    print('d')