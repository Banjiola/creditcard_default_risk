from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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