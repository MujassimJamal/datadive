import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_friedman1
from sklearn.linear_model import LinearRegression

# global variables
X_train_linear_friedman = []
X_test_linear_friedman = []
y_train_linear_friedman = []
y_test_linear_friedman = []
train_score_linear_friedman = None
test_score_linear_friedman = None

# attributes
global_coefficients = []
global_intercept = None
global_features = None

y_actual_label = None
y_predicted_label = None

#model
linear_friedman_classifier = None

heat_friedman = None
friedman_type = None

def Linear(this_noise = 0.0, data_size=None):
    global X_train_linear_friedman, X_test_linear_friedman, y_train_linear_friedman, y_test_linear_friedman
    global train_score_linear_friedman, test_score_linear_friedman, y_actual_label, linear_friedman_classifier, y_predicted_label
    global heat_friedman, friedman_type, global_coefficients, global_intercept, global_features

    Features, Labels = make_friedman1(n_samples=1000, n_features=5, noise=this_noise, random_state=0)   

    df_X = pd.DataFrame(Features, columns=['X1', 'X2', 'X3', 'X4', 'X5'])
    df_y = pd.DataFrame(Labels, columns=['y'])

    heat_friedman = df_X
    friedman_type = ['X1', 'X2', 'X3', 'X4', 'X5']
    y_actual_label = df_y

    X_train_friedman, X_test_friedman, y_train_friedman, y_test_friedman = train_test_split(df_X, df_y, train_size=data_size, random_state=0)
    
    X_train_linear_friedman = X_train_friedman
    X_test_linear_friedman = X_test_friedman
    y_train_linear_friedman = y_train_friedman
    y_test_linear_friedman = y_test_friedman
    
    friedman_clf = LinearRegression().fit(X_train_friedman ,y_train_friedman)
    y_pred = friedman_clf.predict(X_test_friedman)
    y_predicted_label = y_pred
    
    train_score_linear_friedman = "{:.2f}".format(friedman_clf.score(X_train_friedman, y_train_friedman))
    test_score_linear_friedman = "{:.2f}".format(friedman_clf.score(X_test_friedman, y_test_friedman))

    global_coefficients =  [round(i,2) for i in friedman_clf.coef_[0]]
    global_intercept = round(friedman_clf.intercept_[0], 2)
    global_features = friedman_clf.n_features_in_

    linear_friedman_classifier = friedman_clf
