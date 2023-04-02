#-----------------------------------------------------------------------
# This code helps to import relative modules e.g packages from another #
# directory                                                            #
import sys                                                             #
from os import path                                                    # 
sys.path.append( path.dirname(path.dirname( path.abspath(__file__))))  #
#-----------------------------------------------------------------------

# Logistic on Fruit and Cancer

from datasets.cancer import *
from datasets.fruit import *

# Data Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression

# global variables
X_train_logistic_fruit = []
X_test_logistic_fruit = []
y_train_logistic_fruit = []
y_test_logistic_fruit = []
train_score_logistic_fruit = None
test_score_logistic_fruit = None

X_train_logistic_cancer = []
X_test_logistic_cancer = []
y_train_logistic_cancer = []
y_test_logistic_cancer = []
train_score_logistic_cancer = None
test_score_logistic_cancer = None

#model
logistic_fruit_classifier = None
logistic_cancer_classifier = None

# SVC 
def Logistic(this_C, this_penalty, data_size):
    #initializing global variables
    global X_train_logistic_fruit , X_test_logistic_fruit, y_train_logistic_fruit, y_test_logistic_fruit, train_score_logistic_fruit, test_score_logistic_fruit, logistic_classifier
    global X_train_logistic_cancer, X_test_logistic_cancer, y_train_logistic_cancer, y_test_logistic_cancer, train_score_logistic_cancer, test_score_logistic_cancer
    global logistic_fruit_classifier, logistic_cancer_classifier

    # Feature extraction
    X_fruit = df_fruit()[['mass','width', 'height','color_score']]
    y_fruit = df_fruit()['fruit_label']

    # df_cancer already loaded in at line 8.
    X_cancer = df_cancer().iloc[:, 2:]  # excluding id and diagnosis column.
    y_cancer = df_cancer()['diagnosis']

    X_train_fruit, X_test_fruit, y_train_fruit, y_test_fruit = train_test_split(X_fruit, y_fruit, train_size=data_size, random_state=0)
    X_train_cancer, X_test_cancer,y_train_cancer, y_test_cancer = train_test_split(X_cancer, y_cancer, train_size=data_size, random_state=0)

    # feature scaling
    scaler = MinMaxScaler()
    X_train_fruit_scaled = scaler.fit_transform(X_train_fruit)
    X_test_fruit_scaled = scaler.transform(X_test_fruit)

    X_train_cancer_scaled = scaler.fit_transform(X_train_cancer)
    X_test_cancer_scaled = scaler.transform(X_test_cancer) 

    # pca to reduce dimentionality of data
    pca = IncrementalPCA(n_components = 2)
    X_train_fruit_pca = pca.fit_transform(X_train_fruit_scaled)
    X_test_fruit_pca = pca.transform(X_test_fruit_scaled)

    X_train_cancer_pca = pca.fit_transform(X_train_cancer_scaled)
    X_test_cancer_pca = pca.transform(X_test_cancer_scaled)
    
    # Assign PCA sets to global variables.
    X_train_logistic_fruit = X_train_fruit_pca
    X_test_logistic_fruit = X_test_fruit_pca
    y_train_logistic_fruit = y_train_fruit
    y_test_logistic_fruit = y_test_fruit

    X_train_logistic_cancer = X_train_cancer_pca
    X_test_logistic_cancer = X_test_cancer_pca
    y_train_logistic_cancer = y_train_cancer
    y_test_logistic_cancer = y_test_cancer
    
    
    fruit_clf = LogisticRegression(C=this_C, penalty=this_penalty, solver='saga').fit(X_train_fruit_pca, y_train_fruit)
    cancer_clf = LogisticRegression(C=this_C, penalty=this_penalty, solver='saga').fit(X_train_cancer_pca, y_train_cancer)

    train_score_logistic_fruit = "{:.0f}".format(fruit_clf.score(X_train_fruit_pca, y_train_fruit)*100)
    test_score_logistic_fruit = "{:.0f}".format(fruit_clf.score(X_test_fruit_pca, y_test_fruit)*100)

    train_score_logistic_cancer = "{:.0f}".format(cancer_clf.score(X_train_cancer_pca, y_train_cancer)*100)
    test_score_logistic_cancer = "{:.0f}".format(cancer_clf.score(X_test_cancer_pca, y_test_cancer)*100)
    
    logistic_fruit_classifier = fruit_clf
    logistic_cancer_classifier = cancer_clf