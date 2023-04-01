#-----------------------------------------------------------------------
# This code helps to import relative modules e.g packages from another #
# directory                                                            #
import sys                                                             #
from os import path                                                    # 
sys.path.append(path.dirname(path.dirname( path.abspath(__file__))))   #
#----------------------------------------------------------------------#

from algorithms import svm
from algorithms import logistic
from algorithms import linear

import pickle
import os

# saving model as a pickle
def make_pickles():
    path = os.path.join(os.path.dirname(__file__), 'svc_fruit_model.sav')
    pickle.dump(svm.svc_fruit_classifier, open(path, "wb"))

    path = os.path.join(os.path.dirname(__file__), 'logistic_fruit_model.sav')
    pickle.dump(logistic.logistic_fruit_classifier, open(path, "wb"))

    path = os.path.join(os.path.dirname(__file__), 'svc_cancer_model.sav')
    pickle.dump(svm.svc_cancer_classifier, open(path, "wb"))
    
    path = os.path.join(os.path.dirname(__file__), 'logistic_cancer_model.sav')
    pickle.dump(logistic.logistic_cancer_classifier, open(path, "wb"))

    path = os.path.join(os.path.dirname(__file__), 'linear_friedman_model.sav')
    pickle.dump(linear.linear_friedman_classifier, open(path, "wb"))
