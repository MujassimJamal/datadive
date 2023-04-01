#-----------------------------------------------------------------------
# This code helps to import relative modules e.g packages from another #
# directory                                                            #
import sys                                                             #
from os import path                                                    # 
sys.path.append(path.dirname(path.dirname( path.abspath(__file__))))   #
#----------------------------------------------------------------------#

# SVC on Fruit and Cancer
# datasets folder.
from datasets.fruit import *
from datasets.cancer import *

# Data Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA
from sklearn.svm import SVC

# global variables
X_train_svc_fruit = []
X_test_svc_fruit = []
y_train_svc_fruit = []
y_test_svc_fruit = []
train_score_svc_fruit = None
test_score_svc_fruit = None

X_train_svc_cancer = []
X_test_svc_cancer = []
y_train_svc_cancer = []
y_test_svc_cancer = []
train_score_svc_cancer = None
test_score_svc_cancer = None

#model
svc_fruit_classifier = None
svc_cancer_classifier = None

# SVC 
def Svc(this_C, this_kernel, data_size):
    #initializing global variables
    global X_train_svc_fruit , X_test_svc_fruit, y_train_svc_fruit, y_test_svc_fruit, train_score_svc_fruit, test_score_svc_fruit, svc_classifier
    global X_train_svc_cancer, X_test_svc_cancer, y_train_svc_cancer, y_test_svc_cancer, train_score_svc_cancer, test_score_svc_cancer
    global svc_fruit_classifier, svc_cancer_classifier

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
    X_train_svc_fruit = X_train_fruit_pca
    X_test_svc_fruit = X_test_fruit_pca
    y_train_svc_fruit = y_train_fruit
    y_test_svc_fruit = y_test_fruit

    X_train_svc_cancer = X_train_cancer_pca
    X_test_svc_cancer = X_test_cancer_pca
    y_train_svc_cancer = y_train_cancer
    y_test_svc_cancer = y_test_cancer
    
    fruit_clf = SVC(C= this_C, kernel=this_kernel).fit(X_train_fruit_pca, y_train_fruit)
    cancer_clf = SVC(C= this_C, kernel=this_kernel).fit(X_train_cancer_pca, y_train_cancer)

    train_score_svc_fruit = "{:.0f}".format(fruit_clf.score(X_train_fruit_pca, y_train_fruit)*100)
    test_score_svc_fruit = "{:.0f}".format(fruit_clf.score(X_test_fruit_pca, y_test_fruit)*100)

    train_score_svc_cancer = "{:.0f}".format(cancer_clf.score(X_train_cancer_pca, y_train_cancer)*100)
    test_score_svc_cancer = "{:.0f}".format(cancer_clf.score(X_test_cancer_pca, y_test_cancer)*100)
    
    svc_fruit_classifier = fruit_clf
    svc_cancer_classifier = cancer_clf