# Data Cleaning and Processing.
import pandas as pd
import os

heat_cancer = None
cancer_type = None

def df_cancer():
    global heat_cancer, cancer_type

    csv_path = os.path.join(os.path.dirname(__file__), 'data/cancer.csv')
    cancer = pd.read_csv(csv_path)
    
    # drop unamed column
    cancer.drop('Unnamed: 32',inplace=True, axis = 1)
    
    # label encoding
    cancer['diagnosis'] = cancer['diagnosis'].map({'B': 0, 'M' : 1})

    cancer_type = ['Benign', 'Malignant']
    heat_cancer = cancer.iloc[:, 2:6] # heatmap data for 4 features

    return cancer
