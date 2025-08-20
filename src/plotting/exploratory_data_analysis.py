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
# Setting random state for reproducibility and Template for Visualisations
random_state= 70 

# Functions for Exploratory Data Analysis
# 1. Boxplot
def plot_boxplot(column, data, title=None, hue=None, by=None, save=False):
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
        plt.savefig(fname = by + 'boxplot', 
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
def plot_histogram(data, column = None, title=None):
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
        plt.savefig(fname = column, 
                dpi=300, 
                bbox_inches='tight',    
                pad_inches=0.1,         
                facecolor='white')
    plt.show()

if __name__ =="__main__":
    from data.data_collection import load_credit_data
    df = load_credit_data()
    plot_boxplot(data= df, column= 'credit_amount', by = 'marital_status',hue='Y',title= 'Distribution of Credit Amount by Marital Status')


    