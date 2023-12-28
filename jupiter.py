#!/usr/bin/env python
# coding: utf-8

# In[16]:


import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[20]:


class Moons:
    def __init__(self, db_path):
        self.db_path = db_path
        self.data = self.load_data()
     
    def load_data(self):
        # Create connection to SQLite database
        database_service = "sqlite"
        database = "data/jupiter.db"

        connectable = f"{database_service}:///{database}"
        
        query = "SELECT * FROM moons"
        df = pd.read_sql(query, connectable)
        
        return df
    
    def summary_statistics(self):
        return self.data.describe()
    
    def completeness(self):
        return self.data.isnull().sum()
    
    def correlation(self, visualise = False):
        corr_matrix = self.data.corr()
        if visualise:
            sns.heatmap(corr_matrix, annot = True, cmap= 'RdYlGn')
            plt.show()
        return corr_matrix
    
    def get_moon_data(self, moon_name):
        f_data = self.data.loc[self.data['moon'] == moon_name]
        
        # Check if dataframe is empty
        if f_data.empty:
            print(f"No data obtained for the moon {moon_name}.")
        else:
            return f_data
  
    
moons_df = Moons('data/jupiter.db')



