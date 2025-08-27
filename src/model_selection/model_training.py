import pandas as pd
#  Import Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib

random_state= 70 
# split data
def get_train_test_data(X, y, train_size=0.7, random_state=random_state, stratify=False): 
    # using stratify=False is just a design flaw imo because i spent so much time trying 
    # to check how my data changed when the only issue was that i didnt enable stratify. 
    # Moving forward, simply stratify=True
    """
    Split data into training and testing sets with optional stratification.
    
    Parameters
    ----------
    X : pd.DataFrame, array-like
        Feature Matrix.
    y : pd.Series, array-like
        Target Column.
    train_size : float, default=0.7
        Proportion of dataset to include in train split.
    random_state : int, default=random_state
        Random state for reproducibility.
    stratify : bool, default=False
        Whether to stratify the split based on target variable.
    
    Returns
    -------
    X_train, X_test, y_train, y_test : array-like
        Split datasets.
    """
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        stratify=stratify_param, 
        train_size=train_size, 
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test


import joblib
from pathlib import Path

def save_model(model, model_name):
    """Saves a model to the models directory"""
    # Ensure models folder exists
    folder = Path("models")
    folder.mkdir(parents=True, exist_ok=True) # moving forward i should have like a file for constants such as this

    # Full path
    model_path = folder / f"{model_name}.joblib"
    
    # Save the model
    joblib.dump(model, model_path)
    print(f"{model_name} saved at {model_path}") # remove later
    return f"{model_name} saved at {model_path}"


def load_model(model_name):
    """Load a saved model from the models directory"""
    model_path = Path("models") / f"{model_name}.joblib"
    
    # Load the model
    model = joblib.load(model_path)
    print(f"{model_name} loaded from {model_path}") # remove later
    return model


    # Model 1
def train_decision_tree(X_train,y_train, model_name, save=True):
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train,y_train)
    if save:
        save_model(model,model_name)
    return model

# Model 2
def train_logistic_regression(X_train,y_train,model_name,save=True):
    model = LogisticRegression(C= 30, random_state= random_state)
    model.fit(X_train,y_train)
    if save:
        save_model(model,model_name)
    return model

# Model 3
def train_svm(X_train,y_train,model_name,save=True):
    model = SVC(kernel='sigmoid',
                probability= True,
                random_state=random_state)
    model.fit(X_train,y_train)
    if save:
        save_model(model,model_name)
    return model

# Model 4
def train_knn(X_train,y_train,model_name,save=True):
    model = KNeighborsClassifier(n_neighbors= 145)
    model.fit(X_train,y_train)
    if save:
        save_model(model,model_name)

    return model

# Model 5
def train_xgb(X_train,y_train,model_name,save=True):
    model = XGBClassifier()
    model.fit(X_train,y_train)
    if save:
        save_model(model,model_name)
    return model
# Since I am repeating code, I am sure there is a more maintainable way. OOP?

if __name__ == "__main__":
    scaled_X_train_with_engineering = pd.read_csv("datasets/scaled_X_train_engineering.csv")
    X_test= pd.read_csv("datasets/X_test.csv")
    y_train= pd.read_csv("datasets/y_train.csv").squeeze()
    y_test = pd.read_csv("datasets/y_test.csv").squeeze()

    train_decision_tree(scaled_X_train_with_engineering,y_train,"dec_tree_with_feature_engineering")
    train_knn(scaled_X_train_with_engineering,y_train,"knn_with_feature_engineering")
    train_svm(scaled_X_train_with_engineering,y_train,"svm_with_feature_engineering")
    train_xgb(scaled_X_train_with_engineering,y_train,"xgb_feature_engineering")
    train_logistic_regression(scaled_X_train_with_engineering,y_train,"log_reg_with_feature_engineering", save=False)
    print("Done")
