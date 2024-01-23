#!/usr/bin/env python
# coding: utf-8

# In[16]:


import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score



# In[20]:


class Moons:
    def __init__(self, db_path):
        """
        Initialise the Moons class with the database path and load the data.
        """
        self.db_path = db_path
        self.data = self.load_data()
     
    def load_data(self):
        """
        Connect to the SQLite database and load the data into a DataFrame.
        """
        # Create connection to SQLite database
        database_service = "sqlite"
        database = "data/jupiter.db"

        connectable = f"{database_service}:///{database}"
        
        query = "SELECT * FROM moons"
        df = pd.read_sql(query, connectable)
        
        return df
    
    def summary_statistics(self):
        """
        Return summary statistics (count, mean, std, etc.) for each column in the dataset.
        """
        return self.data.describe()
    
    def completeness(self):
        """
        Calculate and return the count of missing values in each column of the dataset.
        """
        return self.data.isnull().sum()
    
    
    def correlation(self, visualise = False):
        """
        Calculate the correlation matrix of the dataset and visualise it as a heatmap.
        """
        corr_matrix = self.data.corr()
        if visualise:
            sns.heatmap(corr_matrix, annot = True, cmap= 'PuRd')
            plt.show()
        return corr_matrix
    
    def plot_scatter(self, x_column, y_column, hue=None, title=None, xlabel=None, ylabel=None, data=None):
        """
        Create a scatter plot for specified columns in the dataset.

        """
        if data is None:
            plot_data = self.data
        else:
            plot_data = data

        if x_column not in plot_data.columns or y_column not in plot_data.columns:
            print(f"Columns {x_column} or {y_column} not found in the data.")
            return

        sns.scatterplot(x = x_column, y = y_column, hue = hue, data = plot_data, palette='bright')
        plt.title(title if title else f'{y_column} vs. {x_column}')
        plt.xlabel(xlabel if xlabel else x_column)
        plt.ylabel(ylabel if ylabel else y_column)
        if hue:
            plt.legend(title=hue)
        plt.show()
        
    
    def get_moon_data(self, moon_name):
        """
        Retrieve and return data for a specified moon.
        """
        f_data = self.data.loc[self.data['moon'] == moon_name]
        
        # Check if dataframe is empty
        if f_data.empty:
            print(f"No data obtained for the moon {moon_name}.")
        else:
            return f_data
        
        
    def prepare_regression_data(self):
        """
        Prepare the dataset for regression analysis by adding T^2 and a^3 columns.
        """
        # Conversion constants
        seconds_per_day = 86400
        meters_per_km = 1000

        # Creating T^2 (in seconds^2) and a^3 (in meters^3) columns
        self.data['T2'] = (self.data['period_days'] * seconds_per_day) ** 2
        self.data['a3'] = (self.data['distance_km'] * meters_per_km) ** 3
        
        
    def train_regression_model(self):
        """
        Train a linear regression model to relate a^3 and T^2, evaluating the performance using R² score.
        """

        X = self.data[['a3']]  # Independent variable (a^3)
        y = self.data['T2']    # Dependent variable (T^2)

        # Linear regression model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        # Model evaluation using R² score
        train_r2 = r2_score(y_train, self.model.predict(X_train))
        test_r2 = r2_score(y_test, self.model.predict(X_test))

        return train_r2, test_r2

    
    def estimate_jupiter_mass(self):
        """
        Estimate and return the mass of Jupiter.
        """

        # The regression coefficient 
        coefficient = self.model.coef_[0]
        
        # Gravitational constant in m^3 kg^-1 s^-2
        G = 6.67430e-11  
        
        # Calculate the estimated mass
        estimated_mass = 4 * np.pi**2 / (G * coefficient)
        return estimated_mass
     
  
    






