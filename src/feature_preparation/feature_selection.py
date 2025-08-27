from feature_preparation.feature_engineering import create_payment_consistency, create_total_delays


def drop_correlated_features(df, columns=None):
    df_copy = df.copy()
    if columns is None:
        columns_to_drop = [ 
        'june_bill',
        'may_bill',
        'april_bill', 
        'june_delay', 
        'may_delay',
        'april_delay'
        ]
    df_copy = df.drop(columns=columns_to_drop)
    return df_copy, columns_to_drop

def perform_engineering(data):
    """
    Apply feature engineering and track newly created continuous features.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.

    Returns
    -------
    data_with_engineering : pd.DataFrame
        Dataset with engineered features added.
    added_continuous_columns : list of str
        List of newly created continuous features.
    """
    added_continuous_columns = []

    data_with_engineering = data.copy()
    # moving forward this should be a tuple that will return the data and column name so that silent errors wont pass
    data_with_engineering = create_payment_consistency(data_with_engineering) 
    data_with_engineering = create_total_delays(data_with_engineering)

    # Track only continuous features
    added_continuous_columns.append("payment_consistency")

    return data_with_engineering, added_continuous_columns

def add_continuous_columns(original_continuous_columns, columns_to_add):
    """
    Add new continuous columns to an existing list.

    Parameters
    ----------
    original_continuous_columns : list of str
        Existing list of continuous features.
    columns_to_add : list of str
        New continuous features to add.

    Returns
    -------
    updated_continuous_columns : list of str
        Updated list of continuous features.
    """
    updated_continuous_columns = original_continuous_columns.copy()
    updated_continuous_columns.extend(columns_to_add)
    return updated_continuous_columns

def scale_continuous_columns(data_with_engineering, continuous_columns, scaler):
    """
    Scale continuous columns in the dataset.

    Parameters
    ----------
    data_with_engineering : pd.DataFrame
        Dataset with features.
    continuous_columns : list of str
        Continuous features to scale.
    scaler : object
        Fitted scaler with .transform() method.

    Returns
    -------
    data_with_engineering : pd.DataFrame
        Dataset with scaled continuous features.
    """
    data_with_engineering = data_with_engineering.copy()
    data_with_engineering[continuous_columns] = scaler.transform(data_with_engineering[continuous_columns])
    return data_with_engineering
    

# Feature engineering --> append continuous column -->  Drop correlated features 
# Feature engineering
def feature_transformation_pipeline(data, continuous_columns): #=X_train
    feature_engineering = perform_engineering(data)
    data_with_engineering = feature_engineering[0]
    added_continuous_columns = feature_engineering[1]

# Append continous column name
    continuous_columns = add_continuous_columns(continuous_columns, added_continuous_columns)

# drop correlated features, as you drop correlated features, you need to return it 
    transformed_data= drop_correlated_features(data_with_engineering)[0]
    dropped_columns= drop_correlated_features(data_with_engineering)[1]
    continuous_columns = [col for col in continuous_columns if col not in dropped_columns]
    return transformed_data, continuous_columns
