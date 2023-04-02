# Data Cleaning and Processing.
import pandas as pd
import os

heat_fruit = None
fruit_type = None


def df_fruit():
    global heat_fruit, fruit_type

    csv_path = os.path.join(os.path.dirname(__file__), 'data/fruits.txt')
    fruit = pd.read_csv(csv_path, sep = "\t") 

    # Fetch unique fruit name and data for heatmap
    fruit_type = list(dict(zip(fruit['fruit_label'].unique(), fruit['fruit_name'].unique())).values())
    heat_fruit = fruit.iloc[:, 3:] # 4 features.

    return fruit