import numpy as np
def create_payment_consistency(df, payment_columns=None):  # Fixed typo too
    """
    Creates `payment_consistency` feature.
    
    Returns
    -------
    pd.DataFrame
        New DataFrame with added `payment_consistency` feature
    """
    df_copy = df.copy()  # ✅ Make a copy first
    
    if payment_columns is None:
        payment_columns = [
            'sept_payment', 
            'august_payment', 
            'july_payment', 
            'june_payment', 
            'may_payment', 
            'april_payment'
        ]
    
    df_copy['payment_consistency'] = df_copy[payment_columns].std(axis=1)
    return df_copy  # ✅ Return the copy


def create_total_delays(df, delay_columns=None):
    """
    Creates `total_delays` feature.
    
    Returns
    -------
    pd.DataFrame
        New DataFrame with added `total_delays` feature
    """
    df_copy = df.copy()  # ✅ Make a copy
    
    if delay_columns is None:
        delay_columns = [
            'april_delay', 'may_delay', 'june_delay',
            'july_delay', 'august_delay', 'sept_delay'
        ]
    
    df_copy['total_delays'] = df_copy[delay_columns].sum(axis=1)
    return df_copy


def transform_credit_amount(df, column=None):
    """
    Applies square root transformation to credit amount.
    
    Returns
    -------
    pd.DataFrame
        New DataFrame with transformed credit amount
    """
    df_copy = df.copy()  # ✅ Make a copy
    
    if column is None:
        column = "credit_amount"
    
    df_copy[column] = np.sqrt(df_copy[column]+1)
    return df_copy
# now i have to normalise the features
from sklearn.preprocessing import StandardScaler
def get_scaler(train_data, columns):
    # Instantiait scaler
    scaler = StandardScaler()
    scaler.fit(X=train_data[columns])
    return scaler

def scale_data(data, columns, scaler):
    data_copy = data.copy()
    data_copy[columns] = scaler.transform(data_copy[columns])
    return data_copy

