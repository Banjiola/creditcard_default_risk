# be careful not to edit the actual data always work on copies or use inplace method if copying is expensive
import numpy as np
def create_payment_consistency(df, payment_columns=None):
    """
    Creates `payment_consistency` feature.
    
    Returns
    -------
    pd.DataFrame
        New DataFrame with added `payment_consistency` feature
    """
    df_copy = df.copy()
    
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
    return df_copy


def create_total_delays(df, delay_columns=None):
    """
    Creates `total_delays` feature.
    
    Returns
    -------
    pd.DataFrame
        New DataFrame with added `total_delays` feature
    """
    df_copy = df.copy()
    
    if delay_columns is None:
        delay_columns = [
            'april_delay', 'may_delay', 'june_delay',
            'july_delay', 'august_delay', 'sept_delay'
        ]
    
    df_copy['total_delays'] = df_copy[delay_columns].sum(axis=1)
    return df_copy

# Functions to scale the data
from sklearn.preprocessing import StandardScaler

def get_scaler(train_data, columns):
    """
    Initialise scaler and fit on train data. This prevents data leakage.

    Returns
    -------
    StandardScaler Object
        fitted scaler on train_data
    """
    scaler = StandardScaler()
    scaler.fit(X=train_data[columns])
    return scaler

def scale_data(data, columns, scaler):
    """Scales the `columns` of selected `data` using scaler.
    
    Returns
    -------
    pd.DataFrame
        Scaled data

    """
    data_copy = data.copy()
    data_copy[columns] = scaler.transform(data_copy[columns])
    return data_copy

