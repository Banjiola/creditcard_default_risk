from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from model_selection.evaluation_utils import get_positive_class_metric
from model_selection.model_training import load_model
from model_selection.model_tuning import load_random_search_object

# setting general template
sns.set_palette("colorblind")
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 11
})

def plot_confusion_matrix(y_pred, y_true, title=None):
    """
    Plots confusion matrix of our model's predictions.
    
    Parameters
    ----------
    y_pred : np.ndarray
        Prediction of the estimator/classifier.
    y_true : pd.Series, optional
        Test data values. 
    title : str, optional
        Title for plot. Defaults to None.

    Returns
    -------
    None
        Simply displays the confusion matrix of our classifier.
    """

    labels = ['No Default', 'Default']

    # We create a dataframe with the appropriate labels
    cm = confusion_matrix(y_true= y_true, y_pred = y_pred)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(6.5, 4))
    sns.heatmap(cm_df, annot=True, fmt='d', cbar= False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    plt.show()

  # models {"name":model,}
  
  
    # I need to do for the train and then for the validation splits
def plot_positive_class_metric(models,X_true, y_true,title, cross_val=False, save=False):
    """
    model: dict
        Dictionary containing model name and actual model.
    
    X_true : array-like
        Feature matrix to train or evaluate the classifier. 

    y_true : array-like
        Target labels corresponding to `X_true`.

    title: str
        Title of Plot and how it will be saved
    
    cross_val : bool, optional, default=False
        Whether to perform cross-validation. If True, uses `cross_val_score`.
    
    cv : StratifiedKfold, optional
    Should be defined globally before passing into the function. 
    Comes into play ` if `cross_val=True`. Passed to `cross_val_score.
    
    """
    recall_result = []
    precision_result = []
    f1_result = []
    model_name=[]
    for name, model in models.items():
        metric = get_positive_class_metric(clf = model,X_true=X_true,
                                            y_true=y_true, cross_val = cross_val) 
        
        recall_result.append(metric[0])
        precision_result.append(metric[1])
        f1_result.append(metric[2])
        model_name.append(name)
        
    all_positive_metrics = pd.DataFrame({
    'Models': model_name,
    'Recall (1)':recall_result,
    "Precision (1)" : precision_result, 
    "F1 score (1)" : f1_result
    })
    all_positive_metrics.sort_values(by= 'Models', inplace = True)

    # We need to reshape the DataFrame into a  long format for grouped bar chart
    all_positive_metrics_reshaped = pd.melt(
    all_positive_metrics,
    id_vars='Models', # type: ignore
    value_vars=['Recall (1)', 'Precision (1)',"F1 score (1)"],
    var_name='Metric',
    value_name='Score'
    )

# Plot grouped bar chart
    plt.figure(figsize=(9.5, 6))
    ax = sns.barplot(data=all_positive_metrics_reshaped, x='Models', y='Score', hue='Metric')

# Add data labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge')

    plt.title(title)
    plt.xticks(rotation=22.5)
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.legend(title='Metric', frameon = True)
    plt.tight_layout()
    if save:
        plt.savefig(f'reports/figures/{title}.png', 
            dpi=300, 
            bbox_inches='tight',    
            pad_inches=0.1,         
            facecolor='white') 
    plt.show()

# not used in main report
def plot_pr_auc(clf, X_train, y_train, model_name, save = False):
    """
    Plot Precision-Recall curve and returns PR-AUC using cross-validation predictions.
   
    Parameters
    ----------
    clf : estimator
        The classifier to evaluate
    X_train: pd.DataFrame
    y_train: pd.Series
    model_name : str
        Name of the model for display purposes.

    Returns
    -------
    float
        PR-AUC score
    """
    # Get probabilities for class 1 (Defaulters)
    y_scores = cross_val_predict(clf, X_train, y_train, cv=5, method='predict_proba')[:, 1]
    precision, recall, tholds = precision_recall_curve(y_train, y_scores)
    pr_auc = average_precision_score(y_train, y_scores)

    # Plot the PR curve
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"{model_name} (PR-AUC = {pr_auc:.3f})", color='blue', linewidth=2)
    plt.axhline(y=y_train.mean(), linestyle='--', color='gray', label='Random Classifier')
    plt.xlabel( "Recall")
    plt.ylabel( "Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend( frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(f"reports/figures/{model_name +" pr-auc"}.png", 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.1,        
                facecolor='white')     
    plt.show()

    print(f"{model_name} PR-AUC: {pr_auc:.3f}")
    return pr_auc

def plot_auc(clf, X_train, y_train, model_name, save= False):
    """
    Plot ROC-AUC curve and returns ROC using cross-validation predictions.
   
    Parameters
    ----------
    clf : estimator
        The classifier to evaluate
    X_train: pd.DataFrame
    y_train: pd.Series
    model_name : str
        Name of the model for display purposes.
                
    Returns
    -------
    float
        PR-AUC score
    """
    # Get probabilities for class 1 (Defaulters)
    y_scores = cross_val_predict(clf, X_train, y_train, cv=5, method='predict_proba')[:, 1]
    fpr, tpr, tholds = roc_curve(y_train, y_scores)

    # Compute AUC
    auc = roc_auc_score(y_train, y_scores)
    # Plot the ROC curve
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})", color='blue', linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(f"reports/figures/{model_name +" auc"}.png", 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.1,        
                facecolor='white')     
    plt.show()

    print(f"{model_name} AUC-ROC: {auc:.3f}")
    return auc

if __name__ =='__main__':
    # load models
       # Load data
    scaled_X_train_with_engineering = pd.read_csv("datasets/scaled_X_train_engineering.csv")
    X_test= pd.read_csv("datasets/X_test.csv")
    y_train= pd.read_csv("datasets/y_train.csv").squeeze()
    y_test = pd.read_csv("datasets/y_test.csv").squeeze()
    
    # load untuned models trained on scaled_X_train_engineering
    tree = load_model("dec_tree_with_feature_engineering")
    log_reg = load_model("log_reg_with_feature_engineering")
    svm = load_model("svm_with_feature_engineering")
    knn = load_model("knn_with_feature_engineering")
    xgb = load_model("xgb_feature_engineering")

    #load tuned_random_search objects
    tree_random_search = load_random_search_object("tree_random_search")
    log_reg_random_search = load_random_search_object("log_reg_random_search")
    svm_random_search = load_random_search_object("svm_random_search")
    knn_random_search = load_random_search_object("knn_random_search")
    xgb_random_search = load_random_search_object("xgb_random_search")

    # EXTRACT MODELS FROM SEARCH OBJECTS
    tuned_tree = tree_random_search[0]
    tuned_log_reg = log_reg_random_search[0]
    tuned_svm = svm_random_search[0]
    tuned_knn = knn_random_search[0]
    tuned_xgb = xgb_random_search[0]
    
    print("loading of random search complete")

    untuned_models = {
    "Decision Tree": tree,
    "Logistic Regression":log_reg,
    "Support Vector Machine":svm,
    "K-Nearest Neighbour":knn,
    "XGBoost":xgb}

    
    tuned_models = {
    "Decision Tree": tuned_tree,
    "Logistic Regression":tuned_log_reg,
    "Support Vector Machine":tuned_svm,
    "K-Nearest Neighbour":tuned_knn,
    "XGBoost":tuned_xgb}

    
    # Evaluation Metrics for UNTUNED Models
    plot_positive_class_metric(
            models=untuned_models,
            X_true=scaled_X_train_with_engineering,
            y_true=y_train,
            save= True,
            cross_val= False,
            title= "Positive Metrics for Untuned Models on Training Set")

    plot_positive_class_metric(
        models=untuned_models,
        X_true=scaled_X_train_with_engineering,
        y_true=y_train,
        save= True,
        cross_val=True,
        title= "Positive Metrics for Untuned Models on Cross-Validation set")

    print("# Evaluation Metrics for UNTUNED Models")

    # Evaluation Metrics for TUNED Models
    plot_positive_class_metric(
            models=tuned_models,
            X_true=scaled_X_train_with_engineering,
            y_true=y_train,
            save= True,
            cross_val=False,
            title= "Evaluation Metrics on Training Set After Tuning (RandomisedSearchCV)")


    plot_positive_class_metric(
        models=tuned_models,
        X_true=scaled_X_train_with_engineering,
        y_true=y_train,
        save= True,
        cross_val=True,
        title= "Evaluation Metrics (positive class) on Cross Validation Splits After Tuning (RandomisedSearchCV)"
        )
    
    print("# Evaluation Metrics for TUNED Models")
    # plots auc curve
    for model_name, tuned_model in tuned_models.items():
        plot_auc(
            clf=tuned_model,
            X_train=scaled_X_train_with_engineering,
            y_train= y_train,
            model_name=model_name,
            save= True)

    # plot pr auc curve
    for model_name, tuned_model in tuned_models.items():
        plot_pr_auc(
            clf=tuned_model,
            X_train=scaled_X_train_with_engineering,
            y_train= y_train,
            model_name=model_name,
            save= True)