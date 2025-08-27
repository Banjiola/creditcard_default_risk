import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data.data_preparation import split_columns

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
# Setting random state for reproducibility and Template for Visualisations
random_state= 70 
# Creation of categorical map for categorical data for easy understanding of the dataset
gender_map = {
    1: 'male',
    2: 'female'
    }

education_map = {
    0: 'others',
    1: 'graduate school',
    2: 'university',
    3: 'high school',
    4: 'others',
    5: 'others',
    6: 'others'
}

marital_map = {
    0: 'others',
    1: 'married',
    2: 'single',
    3: 'divorce'
            }

target_map = {
    0: 'non-default',
    1: 'default'
    }

# Payment Delay Map
delay_map = {
-2: 'no usage',
-1: 'paid full',
0: 'revolving',
1: '1m delay',
2: '2m delay',
3: '3m delay',
4: '4m delay',
5: '5m delay',
6: '6m delay',
7: '7m delay',
8: '8m delay',
9: '9m+ delay'
}

# function to group ages into decades
def convert_age(age):
    """
    Converts age into decades for easy exploration.
    
    Parameters
    ----------
    age : int
        Age of consumer.


    Returns
    -------
    decade : str
        Age converted into relevant decade.
    
    """
    
    # divide age by 10 to determine the decade
    age = age//10
    decade = f'{age*10}s'
    return decade

# Functions for Exploratory Data Analysis
# 1. Boxplot
def plot_boxplot(column, data, title='', hue=None, by=None, save=False):
    """
    Plots the distribution of a numerical feature using a boxplot, including
    median, quartiles, and outliers. Optionally, compares distributions across
    categories using a hue (e.g., distribution of credit_amount by age group).

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the feature and optional hue column.
    column : str
        The name of the numerical column to be plotted (continuous).
    title : str, optional
        The title of the boxplot. Default is None.
    hue : str, optional
        The categorical column to compare distributions across. Default is None.
    by : str, optional
        Categorical column which we eant to show our continuous variable by. Default is None.

    Returns
    -------
    None
        Displays the boxplot visualization.
    """

    
    plt.figure(figsize=(6,4))
    if hue is None:
        sns.boxplot(data= data, y = column, x = by)
    else:
        sns.boxplot(data= data, y = column, x = by, hue = hue)
        plt.legend(frameon='True')

        
    plt.ticklabel_format(style='plain', axis='y') # this disables the scientific notation
    plt.xticks(rotation =22.5)
    plt.title(title)
    plt.tight_layout()     
    if save:
        plt.savefig(fname = f"reports/EDA/{by + 'boxplot'}.png", 
                dpi=300, 
                bbox_inches='tight',    
                pad_inches=0.1,         
                facecolor='white')
    plt.show()




#2. Histogram
'''
Histograms are used to show data density and the overall shape of a distribution.
They help us understand where values cluster (high density), where they’re rare (low density)
'''
def plot_histogram(data, column = None, title=''):
    """
    Shows the data distribution and shape of distribution through a histogram for feature/target variable.

    Parameters
    ----------
    data : pd.DataFrame
        Data containing target and feature.
    column : str
        Name of column to be plotted.
    title : str
        Title of the histogram.

    Returns
    -------
    None
        Returns the histogram.
    """

    if column in data:
        plt.figure(figsize= (7,4.5))
        sns.histplot(data= data, x = column, kde= True, label = column, color= 'blue')
        plt.ticklabel_format(style='plain', axis='x')
        plt.legend(frameon= True)
        plt.title(title)
        plt.grid(True, alpha = 0.3, axis= 'both')
        plt.tight_layout()
        plt.show()
    else:
        print(f"{column} not in data")

#3. Barchart
def plot_barchart(column, data, x_label, title=None, hue=None, rotation=0, save = False):
    """
    This function creates a count plot (bar chart) to visualize the frequency
    distribution of categories in the specified column.

    Parameters
    ----------
    column : str
        Name of the column to plot.
    title : str
        Title for the plot.
    x_label : str
        Label for the x-axis.
    data : pd.DataFrame, optional
        The dataset to use.
    rotation : float, optional
        The angle to rotate the xticks. Default is 0.
    save: bool, defaults to False
        Gives Direction on whether to save or not.
    Returns
    -------
    None
        Displays the bar chart but returns nothing.
    """

    plt.figure(figsize=(6,4))
    ax = sns.countplot(data= data, x= column, hue = hue) # add hue = target variable

    for p in ax.patches:
        count = int(p.get_height())
        ax.annotate(
            str(count), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Count")
    plt.xticks(rotation = rotation)
    
    if hue is not None:
        plt.legend(frameon = True)

    plt.tight_layout()
    if save:
        plt.savefig(fname = f"reports/EDA/{column}.png", 
                    dpi=300, 
                bbox_inches='tight',    
                pad_inches=0.1,         
                facecolor='white')
    plt.show()

if __name__ =="__main__":
    from data.data_collection import load_credit_data
    df = load_credit_data()
    df_eda = df.copy()

    # we will need to split the columns into categorical and continuous
    splitted_columns = split_columns(df)
    categorical_columns = splitted_columns['categorical_columns']
    continuous_columns = splitted_columns['continuous_columns']
    
    # let us remove age from continuous and append to categorical since i want to kinda bin it
    continuous_columns.remove('age')
    categorical_columns.append('age')

    
    print("Five rows for categorical data (before mapping)".center(100,'='))
    print(df_eda[categorical_columns].head())

    # application of the categorical map to all the columns
    df_eda['gender'] = df_eda['gender'].map(gender_map )
    df_eda['education'] = df_eda['education'].map(education_map)
    df_eda['marital_status'] = df_eda['marital_status'].map(marital_map)
    df_eda['Y'] = df_eda['Y'].map(target_map)
    df_eda['age'] = df_eda['age'].map(convert_age)

    # Select delay columns
    delay = [column_name for column_name in categorical_columns if 'delay' in column_name.lower()]
    df_eda[delay] = df_eda[delay].apply(lambda column: column.map(delay_map))

    # let us check the categorical data now
    print("Five rows for categorical data (after mapping)".center(100,'='))
    print(df_eda[categorical_columns].head())

    bill = [column_name for column_name in continuous_columns if 'bill' in column_name]
    payment = [column_name for column_name in continuous_columns if 'payment' in column_name]

    # Create total bill and payment for each consumer
    df_eda["total_bill"] = df_eda[bill].sum(axis=1) 
    df_eda['total_payment'] = df_eda[payment].sum(axis=1)

    df_eda.head()

    plot_boxplot(data= df_eda, column= 'credit_amount', by = 'marital_status',hue='Y',title= 'Distribution of Credit Amount by Marital Status', save=True)
    plot_boxplot(data= df_eda, column= 'credit_amount', by = 'education',hue='Y',title = 'Distribution of Credit Amount by Education', save=True)
    plot_boxplot(data= df_eda, column= 'credit_amount', by = 'gender',hue='Y', title = 'Distribution of Credit Amount by Gender', save=True)
    plot_boxplot(data= df_eda, column= 'credit_amount', by = 'Y', title = 'Distribution of Credit Amount by Default', save=True)
    plot_barchart(data=df_eda,column= 'age', title= 'Distribution of Age', x_label="Age Category", save=True )

    # Barchart by Default Status
    plot_barchart(data=df_eda, column= 'age', title= 'Number of loan applicants by Age and Default status', hue = 'Y', x_label="Age Category", save=True ) #hue = 'Y',
    plot_barchart(data=df_eda,column= 'marital_status', title= 'Number of loan applicants by Marital and Default Status', hue = 'Y',x_label="Marital Status", save=True )
    plot_barchart(data=df_eda,column= 'gender', title= 'Number of loan applicants by Gender and Default Status', hue = 'Y',x_label="Gender", save=True )
    plot_barchart(data=df_eda,column= 'education', title= 'Number of loan applicants by Education and Default Status', hue = 'Y',x_label="Education", save=True )



    # We use boxplot to compare continuous data by categorical data
    # credit_amount by marital_status # prefer the title 'Distribution of Credit Amount by Marital Status


    plot_barchart(data=df_eda,column= 'marital_status', title= 'Distribution of Marital Status', x_label="Marital Status", save=True )
    plot_barchart(data=df_eda,column= 'gender', title= 'Distribution of Gender', x_label="Gender", save=True )
    plot_barchart(data=df_eda,column= 'education', title= 'Distribution of Education', x_label="Education", save=True )
    plot_barchart(data=df_eda,column= 'Y', title= 'Distribution of Target Variable', x_label="Target Variable", save=True )

    # Delays
    plot_barchart(data=df_eda,column= 'april_delay', title= 'Distribution of April Delay', x_label="April", rotation= 23, save=True)
    plot_barchart(data=df_eda,column= 'may_delay', title= 'Distribution of May Delay', x_label="May", rotation= 23, save=True)
    plot_barchart(data=df_eda,column= 'june_delay', title= 'Distribution of June Delay', x_label="June", rotation= 23, save=True)
    plot_barchart(data=df_eda,column= 'july_delay', title= 'Distribution of July Delay', x_label="July", rotation= 23, save=True)
    plot_barchart(data=df_eda,column= 'august_delay', title= 'Distribution of August Delay', x_label="August", rotation= 23, save=True)
    plot_barchart(data=df_eda,column= 'sept_delay', title= 'Distribution of September Delay', x_label="September", rotation= 23, save=True)