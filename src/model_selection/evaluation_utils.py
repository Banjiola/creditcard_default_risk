from sklearn.metrics import classification_report, recall_score, f1_score, precision_score\
    ,precision_recall_curve,average_precision_score, roc_auc_score, roc_curve

from sklearn.model_selection import cross_val_predict
# Setting random state for reproducibility and Template for Visualisations
random_state= 70 

def estimator_report(y_pred, y_true, title=None):
    """
    Parameters
    ----------
    y_pred : np.ndarray
        Prediction of the estimator/classifier.
    y_true : pd.Series, optional
        Test data values.
    title : str, optional
        Title for metrics that would be produced. Defaults to `None`.

    Returns
    -------
    None
        Simply prints some metrics for our classifier.
    """



    print(f"{'='*15} {title} {'='*10}")
    print(f"{'='*20} Classification Report {'='*20}")
    print(classification_report(y_true = y_true, y_pred= y_pred, target_names=['Non-Default', 'Default']))

    print((20+len(' Classification Report ')+20)*'=')
    print(f"Recall: {recall_score(y_true, y_pred):.5f}")
    print(f"Precision-Score: {precision_score(y_true, y_pred):.5f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.5f}")


def get_train_metrics(clf, title, X_train, y_train):
    """
    Gets the model's performance on the full training data (without CV).

    Parameters
    ----------
    - clf : estimator
        sklearn estimator used for the classification. The estimator must not be fitted yet.
    - title : str
        Title for metrics that would be produced.
    - X_train : array-like, optional
        Training Feature Matrix. 
    - y_train : pd.Series, optional
        Train data values.
    
    Returns
    -------
    None
        Outputs classification report metrics and confusion plot.
    """

    clf.fit(X_train,y_train)
    clf_train_pred = clf.predict(X_train) 
    estimator_report(y_pred=clf_train_pred,
                     y_true=y_train,
                     title= title)

def get_cv_metrics(clf, title,X_train, y_train, cv):
    """
    Estimates the model's performance on unseen data using cross-validation on the training set.

    Parameters
    ----------
    - clf : sklearn.estimator
        Sklearn estimator used for the classification. The estimator must not be fitted yet.
    - title : str
        Title for metrics that would be produced.
    - X_train : array-like, optional
        Training Feature Matrix.
    - y_train : pd.Series, optional
        Train data values.
    - cv: StratifiedKfold CV (defined Globally)

    Returns
    -------
    None
        Outputs classifcation report metrics and Confusion Plot.
    """
    # prediction using cross validation
    # cv = 5, we are using stratified kfold
    cv_pred = cross_val_predict(
        clf,
        X= X_train,
        y= y_train,
        cv = cv)

    # Evaluate performance
    estimator_report(y_true=cv_pred,
                    y_pred= y_train,
                    title= title)
# This is specifically used for the getting of evaluation metrics for our models
# which will be used to plot the grouped bar chart for the report.

def get_positive_class_metric(clf, X_true, y_true, cv, cross_val = False):

    """
    Gets the recall, precision, and f1_score of the positive class for classifier's using training data or cross-validation set.
 
    Parameters
    ----------
    clf : estimator object
       Already fitted Scikitlear/XGBoost Estimator. 

    X_true : array-like, optional
        Feature matrix to train or evaluate the classifier. 

    y_true : array-like, optional
        Target labels corresponding to `X_true`.

    cross_val : bool, optional, default=False
        Whether to perform cross-validation. If True, uses `cross_val_score`.

    cv : StratifiedKfold, optional
        Should be defined globally before passing into the function. 
        Comes into play ` if `cross_val=True`. Passed to `cross_val_score.

    Returns
    -------
    tuple
        A tuple containing
        Recall: float
            The recall score of the positive class of the classifier if `cross_val=False`, or the mean
            cross-validation score if `cross_val=True`.
        
        Precision: float
            The precision score of the positive class of the classifier if `cross_val=False`, or the mean
            cross-validation score if `cross_val=True`.

        f1_score: float
            The precision score of the positive class of the classifier if `cross_val=False`, 
            or the mean cross-validation score if `cross_val=True`.
    """


    if cross_val is False:
        clf.fit(X_true,y_true)
        y_pred =  clf.predict(X_true)
    elif cross_val is True:
        y_pred = cross_val_predict(clf,
                                X= X_true,
                                y=y_true,
                                cv = cv)
    else:
        return "Invalid cross_val response"
    
    recall = recall_score(y_true,y_pred, pos_label = 1)
    precision = precision_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label= 1)
    return recall, precision, f1