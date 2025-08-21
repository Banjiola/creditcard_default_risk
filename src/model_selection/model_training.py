# Import Classifiers
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


    # Model 1
def train_decision_tree(X_train,y_train, model_name):
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train,y_train)
    joblib.dump(model,f"../models/{model_name}.joblib")
    print(f"{model_name} complete and saved")
# Model 2
def train_logistic_regression(X_train,y_train,model_name):
    model = LogisticRegression(C= 30, random_state= random_state)
    model.fit(X_train,y_train)
    joblib.dump(model,f"../models/{model_name}.joblib")
    print(f"{model_name} complete and saved")

# Model 3
def train_svm(X_train,y_train,model_name):
    model = SVC(kernel='sigmoid',
                probability= True,
                random_state=random_state)
    model.fit(X_train,y_train)
    joblib.dump(model,f"../models/{model_name}.joblib")
    print(f"{model_name} complete and saved")

# Model 4
def train_knn(X_train,y_train,model_name):
    model = KNeighborsClassifier(n_neighbors= 145)
    model.fit(X_train,y_train)
    joblib.dump(model,f"../models/{model_name}.joblib")
    print(f"{model_name} complete and saved")

# Model 5
def train_xgb(X_train,y_train,model_name):
    model = XGBClassifier()
    model.fit(X_train,y_train)
    joblib.dump(model,f"../models/{model_name}.joblib")
    print(f"{model_name} complete and saved")

# Since I am repeating code, I am sure there is a more maintainable way. OOP?