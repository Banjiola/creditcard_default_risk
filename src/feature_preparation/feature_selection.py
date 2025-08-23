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