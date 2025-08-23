def split_columns(df,continuous_columns=None):
    """
    Splits orginal data into categorical and continuous columns.

    Returns
    -------
    dict
        - continuous_columns (list)
        - categorical_columns (list)
    """
    # Continuous columns
    if continuous_columns is None:
        continuous_columns= [
            'credit_amount',
            'sept_bill',
            'august_bill',
            'july_bill',
            'june_bill',
            'may_bill',
            'april_bill',
            'sept_payment',
            'august_payment',
            'july_payment',
            'june_payment',
            'may_payment',
            'april_payment',
            "age"
            ]
        
    categorical_columns = list(set(df.columns.to_list()) - set(continuous_columns))
    return {"continuous_columns": continuous_columns,
            "categorical_columns": categorical_columns}

def remove_duplicates(df):
    """Removes duplicate values."""
    df = df.drop_duplicates()
    return df

def remove_nulls(df):
    """Drop Null entries"""
    df = df.dropna()
    return df