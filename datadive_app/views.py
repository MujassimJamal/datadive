# #-----------------------------------------------------------------------
# # This code helps to import relative modules e.g packages from another #
# # directory                                                            #
# import sys                                                             #
# from os import path                                                    # 
# sys.path.append(path.dirname(path.dirname( path.abspath(__file__))))   #
# #----------------------------------------------------------------------#

#-----------------------------------
from django.shortcuts import render
#-----------------------------------

from .datasets import fruit
from .datasets import cancer
from .datasets.fruit import *
from .datasets.cancer import *

from .algorithms import svm, logistic, linear
from .algorithms.svm import *
from .algorithms.logistic import *
from .algorithms.linear import *

from .pickles.pickles import *

from .graphs import plot_scatter, plot_heatmap
from .graphs.plot_scatter import *
from .graphs.plot_heatmap import *


#paras and algos
global_problem_type = None
global_graph_type = None
global_algo_type = None
global_kernel_type = None
global_c_para = None
global_penalty = None
global_noise= None
global_show_test = None
global_show_train = None
global_data_size = None
global_data_type = None
global_dataset = None


#home page
def index(request):
    # Initialization with default parameters.
    df_fruit(), df_cancer()
    Svc(this_C=1, this_kernel='rbf', data_size=0.75)
    Logistic(this_C=1, this_penalty='l2', data_size=0.75)
    Linear(this_noise=0.0, data_size=0.75)
    make_pickles()
    heatmap()
    scatter(show_test='on')
    
    # pass stored varibales after initialization
    return render(request, 'index.html',
                  {'fruit_svc_scatter' : plot_scatter.fruit_svc_scatter,
                  'train_score_svc_fruit' : svm.train_score_svc_fruit,
                  'test_score_svc_fruit' : svm.test_score_svc_fruit})


def result(request):
    # Initialize datasets.
    df_fruit(), df_cancer()

    # initialize global variables
    global global_algo_type, global_graph_type, global_problem_type, global_c_para, global_kernel_type, global_penalty, global_data_size
    global global_data_type, global_noise, global_show_test, global_show_train, global_dataset

    # fetch user selected values from FORM
    try:
        dataset = request.GET['dataset']
        global_dataset = dataset

    except :
        if global_dataset is None:
            global_dataset = '/static/images/fruit.png'

    algorithm = request.GET['algorithm']
    graph_type = request.GET['graph']
    problem_type = request.GET['problem']
    slider_value = request.GET['data_size']

    global_algo_type = algorithm
    global_graph_type = graph_type
    global_problem_type = problem_type
    global_data_size = slider_value

    try:
        checkbox = request.GET['show_test']
        global_show_test = checkbox
    except:
        global_show_test = "off"

    try:
        checkbox2 = request.GET['show_train']
        global_show_train = checkbox2
    except :
        global_show_train = "off"

    # initialize algorithms
    if algorithm == "svm":
        try:
            kernel = request.GET['kernel']
            c = float(request.GET['c'])
            global_kernel_type = kernel
            global_c_para = c

            Svc(this_C=c, this_kernel = kernel, data_size=float(slider_value)/100)

            if global_penalty is not None:
                Logistic(this_C=global_c_para, this_penalty=global_penalty, data_size=float(slider_value)/100)
            else:
                Logistic(this_C=global_c_para, this_penalty='l2', data_size=float(slider_value)/100)

            if global_noise is not None:
                Linear(this_noise=global_noise, data_size=float(slider_value)/100)
            else:
                Linear(this_noise=0.0, data_size=float(slider_value)/100)
        
        except:
            global_penalty = request.GET['penalty']

            if global_c_para is not None and global_kernel_type is not None:
                Svc(this_C=global_c_para, this_kernel=global_kernel_type, data_size=float(slider_value)/100)
            else:
                Svc(this_C=global_c_para, this_kernel='rbf', data_size=float(slider_value)/100)

            if global_c_para is not None and global_penalty is not None:
                Logistic(this_C=global_c_para, this_penalty=global_penalty, data_size=float(slider_value)/100)
            else:
                Logistic(this_C=global_c_para, this_penalty='l2', data_size=float(slider_value)/100)

            if global_noise is not None:
                Linear(this_noise=global_noise, data_size=float(slider_value)/100)
            else:
                Linear(this_noise=0.0, data_size=float(slider_value)/100)

    if algorithm == 'logistic':
        try:
            penalty = request.GET['penalty']
            c = float(request.GET['c'])
            global_c_para = c
            global_penalty = penalty
            
            Logistic(this_C=c, this_penalty=penalty, data_size=float(slider_value)/100)

            if global_kernel_type is not None:
                Svc(this_C=global_c_para, this_kernel=global_kernel_type, data_size=float(slider_value)/100)
            else:
                Svc(this_C=global_c_para, this_kernel='rbf', data_size=float(slider_value)/100)

            if global_noise is not None:
                Linear(this_noise=global_noise, data_size=float(slider_value)/100)
            else:
                Linear(this_noise=0.0, data_size=float(slider_value)/100)

        except :
            global_c_para = float(request.GET['c'])
            global_kernel_type = request.GET['kernel']

            if global_penalty is not None and global_c_para is not None:
                Logistic(this_C=global_c_para, this_penalty=global_penalty, data_size=float(slider_value)/100)
            else:
                Logistic(this_C=global_c_para, this_penalty='l2', data_size=float(slider_value)/100)

            if global_c_para is not None and global_kernel_type is not None:
                Svc(this_C=global_c_para, this_kernel=global_kernel_type, data_size=float(slider_value)/100)
            else:
                Svc(this_C=global_c_para, this_kernel='rbf', data_size=float(slider_value)/100)

            if global_noise is not None:
                Linear(this_noise=global_noise, data_size=float(slider_value)/100)
            else:
                Linear(this_noise=0.0, data_size=float(slider_value)/100)

    if problem_type == "regression":
        try:
            noise = float(request.GET['linear_noise'])
            global_noise = noise

            Linear(this_noise=noise, data_size=float(slider_value)/100)

            if global_c_para is not None and global_kernel_type is not None:
                Svc(this_C = global_c_para, this_kernel=global_kernel_type, data_size=float(slider_value)/100)
            else:
                Svc(this_C=global_c_para, this_kernel='rbf', data_size=float(slider_value)/100)

            if global_c_para is not None and global_penalty is not None:
                Logistic(this_C=global_c_para, this_penalty=global_penalty, data_size=float(slider_value)/100)
            else:
                Logistic(this_C=global_c_para, this_penalty='l2', data_size=float(slider_value)/100)
                
        except:
            try:
                global_c_para = int(request.GET['c'])
            except:
                global_c_para = 1.0
            try:
                global_kernel_type = request.GET['kernel']
            except:
                pass

            if global_noise is not None:
                Linear(this_noise=global_noise, data_size=float(slider_value)/100)
            else:
                global_noise = 0
                Linear(this_noise=0.0, data_size=float(slider_value)/100)

            if global_c_para is not None and global_kernel_type is not None: 
                Svc(this_C = global_c_para, this_kernel=global_kernel_type, data_size=float(slider_value)/100)
            else:
                Svc(this_C=global_c_para, this_kernel='rbf', data_size=float(slider_value)/100)

            if global_c_para is not None and global_penalty is not None:
                Logistic(this_C=global_c_para, this_penalty=global_penalty, data_size=float(slider_value)/100)
            else:
                Logistic(this_C=global_c_para, this_penalty='l2', data_size=float(slider_value)/100)

    # Initialize pickles
    make_pickles()
    # Initialize graphs
    heatmap()
    scatter(show_test=global_show_test, show_train=global_show_train)

    
    if global_problem_type == 'regression':
        global_algo_type = 'linear'
    elif global_problem_type == 'classification' and global_algo_type == 'svm':
        global_algo_type = 'svm'
    elif  global_problem_type == 'classification' and global_algo_type == 'logistic':
        global_algo_type = 'logistic'
    else:
        global_algo_type = 'svm'
        
    #============================================================
    # Make dictionary to pass keys and values to frontend.
    
    my_dict = {
        'fruit_svc_scatter' : plot_scatter.fruit_svc_scatter ,'fruit_logistic_scatter' : plot_scatter.fruit_logistic_scatter, 
        'fruit_heatmap' : plot_heatmap.fruit_heatmap,
        'train_score_svc_fruit' : svm.train_score_svc_fruit, 'train_score_logistic_fruit' : logistic.train_score_logistic_fruit,
        'test_score_svc_fruit' : svm.test_score_svc_fruit, 'test_score_logistic_fruit' : logistic.test_score_logistic_fruit,
        
        'cancer_svc_scatter' : plot_scatter.cancer_svc_scatter, 'cancer_logistic_scatter' : plot_scatter.cancer_logistic_scatter,
        'cancer_heatmap' : plot_heatmap.cancer_heatmap,
        'train_score_svc_cancer' : svm.train_score_svc_cancer, 'train_score_logistic_cancer' : logistic.train_score_logistic_cancer,
        'test_score_svc_cancer' : svm.test_score_svc_cancer, 'test_score_logistic_cancer' : logistic.test_score_logistic_cancer,
        
        'friedman_linear_scatter' : plot_scatter.friedman_linear_scatter , 'friedman_heatmap' : plot_heatmap.friedman_heatmap,
        'train_score_linear_friedman' : linear.train_score_linear_friedman, 'test_score_linear_friedman' : linear.test_score_linear_friedman,
        'global_coefficients' : linear.global_coefficients, 'global_intercept' : linear.global_intercept,      
        'global_features' : linear.global_features,
        
        'global_algo_type' : global_algo_type, 'global_graph_type' : global_graph_type, 'global_problem_type' : global_problem_type,
        'global_c_para' : global_c_para, 'global_kernel_type' : global_kernel_type, 'global_penalty' : global_penalty, 
        'global_data_size' : global_data_size, 'global_data_type' : global_data_type, 
        'global_noise' : global_noise, 'global_show_test' : global_show_test, 'global_show_train' : global_show_train,
        'global_dataset' : global_dataset,
    }

    return render(request, 'result.html', my_dict)