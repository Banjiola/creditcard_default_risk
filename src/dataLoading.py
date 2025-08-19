from ucimlrepo import fetch_ucirepo

# fetch dataset from uci
def load_credit_data():
    default_of_credit_card_clients = fetch_ucirepo(id=350)

# data (as pandas dataframes)
    X = default_of_credit_card_clients.data.features
    y = default_of_credit_card_clients.data.targets
# Renaming of columns to intuitive names
    names = {
    'X1': 'credit_amount',
    'X2':'gender',
    'X3': 'education',
    'X4':'marital_status',
    'X5':'age',
    'X6': 'sept_delay',
    'X7':'august_delay',
    'X8':'july_delay',
    'X9':'june_delay',
    'X10':'may_delay',
    'X11':'april_delay',
    'X12': 'sept_bill' ,
    'X13': 'august_bill',
    'X14': 'july_bill'   ,
    'X15': 'june_bill',
    'X16':'may_bill'   ,
    'X17':'april_bill' ,
    'X18':'sept_payment',
    'X19':'august_payment',
    'X20':'july_payment'   ,
    'X21':'june_payment'   ,
    'X22':'may_payment'     ,
    'X23':'april_payment'}
    X = X.copy().rename(columns= names)
    X.head()      
    return X,y

if __name__ == "__main__": 
    X,y = load_credit_data()
    print(X.head())
  