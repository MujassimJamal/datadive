#-----------------------------------------------------------------------
# This code helps to import relative modules e.g file  import from     #
# another directory                                                    #
import sys                                                             #
from os import path                                                    # 
sys.path.append(path.dirname(path.dirname( path.abspath(__file__))))   #
#----------------------------------------------------------------------#

from algorithms import svm, logistic, linear

from algorithms.svm import *
from algorithms.logistic import *
from algorithms.linear import *

from datasets import fruit, cancer

import io
import os
import pickle
import numpy as np
import urllib, base64
import numpy
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn import neighbors
import matplotlib.patches as mpatches
import graphviz
from sklearn.tree import export_graphviz
from pathlib import Path
#--------------------------------------------------------------------------------------

fruit_svc_scatter = None
fruit_logistic_scatter = None
cancer_svc_scatter = None
cancer_logistic_scatter = None
friedman_linear_scatter = None

def scatter(show_test='on', show_train='on'):
    global fruit_svc_scatter, fruit_logistic_scatter, cancer_svc_scatter, cancer_logistic_scatter, friedman_linear_scatter

    # svc scatter for fruit
    numClasses = numpy.amax(svm.y_train_svc_fruit) + 1
    color_list_light = ['#FFFFAA', '#EFEFEF', '#AAFFAA', '#AAAAFF']
    color_list_bold = ['#EEEE00', '#000000', '#00CC00', '#0000CC']
    cmap_light = ListedColormap(color_list_light[0:numClasses])
    cmap_bold  = ListedColormap(color_list_bold[0:numClasses])
    
    h = 0.03
    k = 0.5
    x_plot_adjust = 0.1
    y_plot_adjust = 0.1
    plot_symbol_size = 50
    
    x_min = svm.X_train_svc_fruit[:, 0].min()
    x_max = svm.X_train_svc_fruit[:, 0].max()
    y_min = svm.X_train_svc_fruit[:, 1].min()
    y_max = svm.X_train_svc_fruit[:, 1].max()
    
    x2, y2 = numpy.meshgrid(numpy.arange(x_min-k, x_max+k, h), numpy.arange(y_min-k, y_max+k, h))

    # load saved model
    script_dir = os.path.dirname(__file__) # current file dir
    file = os.path.join(script_dir, '../pickles/svc_fruit_model.sav')
    svc_fruit_model = pickle.load(open(file, "rb"))

    P = svc_fruit_model.predict(numpy.c_[x2.ravel(), y2.ravel()])
    P = P.reshape(x2.shape)
    
    f = plt.figure()
    ax = f.add_subplot(111)
    
    plt.contourf(x2, y2, P, cmap=cmap_light, alpha = 0.8)
    
    if show_test == "on" and show_train == "on":
        plt.scatter(svm.X_train_svc_fruit[:, 0], svm.X_train_svc_fruit[:, 1], c = svm.y_train_svc_fruit, cmap=cmap_bold, s=plot_symbol_size, edgecolor = 'black')
        plt.xlim(x_min - x_plot_adjust, x_max + x_plot_adjust)
        plt.ylim(y_min - y_plot_adjust, y_max + y_plot_adjust)
        plt.scatter(svm.X_test_svc_fruit[:, 0], svm.X_test_svc_fruit[:, 1], c = svm.y_test_svc_fruit, cmap=cmap_bold, s=plot_symbol_size, marker='^', edgecolor = 'black')

    elif show_test == "on" and show_train == "off":
        plt.xlim(x_min - x_plot_adjust, x_max + x_plot_adjust)
        plt.ylim(y_min - y_plot_adjust, y_max + y_plot_adjust)
        plt.scatter(svm.X_test_svc_fruit[:, 0], svm.X_test_svc_fruit[:, 1], c = svm.y_test_svc_fruit, cmap=cmap_bold, s=plot_symbol_size, marker='^', edgecolor = 'black')

    elif show_test == "off" and show_train == "on":
        plt.scatter(svm.X_train_svc_fruit[:, 0], svm.X_train_svc_fruit[:, 1], c = svm.y_train_svc_fruit, cmap=cmap_bold, s=plot_symbol_size, edgecolor = 'black')
        plt.xlim(x_min - x_plot_adjust, x_max + x_plot_adjust)
        plt.ylim(y_min - y_plot_adjust, y_max + y_plot_adjust)

    legend_handles = []
    for i in range(0, len(fruit.fruit_type)):
        patch = mpatches.Patch(color=color_list_bold[i], label = fruit.fruit_type[i])
        legend_handles.append(patch)
    plt.legend(loc=4, handles=legend_handles, handlelength=1, prop={'size': 8})
    
    # set y axis labels to right side.
    ax.yaxis.tick_right()
    
    fig = plt.gcf()
    #convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri =  urllib.parse.quote(string)
    fruit_svc_scatter = uri


    # logistic scatter for fruit
    numClasses = numpy.amax(logistic.y_train_logistic_fruit) + 1
    color_list_light = ['#FFFFAA', '#EFEFEF', '#AAFFAA', '#AAAAFF']
    color_list_bold = ['#EEEE00', '#000000', '#00CC00', '#0000CC']
    cmap_light = ListedColormap(color_list_light[0:numClasses])
    cmap_bold  = ListedColormap(color_list_bold[0:numClasses])
    
    h = 0.03
    k = 0.5
    x_plot_adjust = 0.1
    y_plot_adjust = 0.1
    plot_symbol_size = 50
    
    x_min = logistic.X_train_logistic_fruit[:, 0].min()
    x_max = logistic.X_train_logistic_fruit[:, 0].max()
    y_min = logistic.X_train_logistic_fruit[:, 1].min()
    y_max = logistic.X_train_logistic_fruit[:, 1].max()
    
    x2, y2 = numpy.meshgrid(numpy.arange(x_min-k, x_max+k, h), numpy.arange(y_min-k, y_max+k, h))

    # load saved model
    script_dir = os.path.dirname(__file__) # current file dir
    file = os.path.join(script_dir, '../pickles/logistic_fruit_model.sav')
    logistic_fruit_model = pickle.load(open(file, "rb"))

    P = logistic_fruit_model.predict(numpy.c_[x2.ravel(), y2.ravel()])
    P = P.reshape(x2.shape)
    
    f = plt.figure()
    ax = f.add_subplot(111)
    
    plt.contourf(x2, y2, P, cmap=cmap_light, alpha = 0.8)

    if show_test == "on" and show_train == "on":
        plt.scatter(logistic.X_train_logistic_fruit[:, 0], logistic.X_train_logistic_fruit[:, 1], c=logistic.y_train_logistic_fruit, cmap=cmap_bold, s=plot_symbol_size, edgecolor = 'black')
        plt.xlim(x_min - x_plot_adjust, x_max + x_plot_adjust)
        plt.ylim(y_min - y_plot_adjust, y_max + y_plot_adjust)
        plt.scatter(logistic.X_test_logistic_fruit[:, 0], logistic.X_test_logistic_fruit[:, 1], c=logistic.y_test_logistic_fruit, cmap=cmap_bold, s=plot_symbol_size, marker='^', edgecolor = 'black')

    elif show_test == "on" and show_train == "off":
        plt.xlim(x_min - x_plot_adjust, x_max + x_plot_adjust)
        plt.ylim(y_min - y_plot_adjust, y_max + y_plot_adjust)
        plt.scatter(logistic.X_test_logistic_fruit[:, 0], logistic.X_test_logistic_fruit[:, 1], c=logistic.y_test_logistic_fruit, cmap=cmap_bold, s=plot_symbol_size, marker='^', edgecolor = 'black')

    elif show_test == "off" and show_train == "on":
        plt.scatter(logistic.X_train_logistic_fruit[:, 0], logistic.X_train_logistic_fruit[:, 1], c=logistic.y_train_logistic_fruit, cmap=cmap_bold, s=plot_symbol_size, edgecolor = 'black')
        plt.xlim(x_min - x_plot_adjust, x_max + x_plot_adjust)
        plt.ylim(y_min - y_plot_adjust, y_max + y_plot_adjust)
    
    legend_handles = []
    for i in range(0, len(fruit.fruit_type)):
        patch = mpatches.Patch(color=color_list_bold[i], label=fruit.fruit_type[i])
        legend_handles.append(patch)
    plt.legend(loc=4, handles=legend_handles, handlelength=1, prop={'size': 8})
    
    # set y axis labels to right side.
    ax.yaxis.tick_right()
    
    fig = plt.gcf()
    #convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri =  urllib.parse.quote(string)
    fruit_logistic_scatter = uri


    # svc scatter for cancer
    numClasses = numpy.amax(svm.y_train_svc_cancer) + 1
    color_list_light = ['#FFFFAA', '#EFEFEF', '#AAFFAA', '#AAAAFF']
    color_list_bold = ['#EEEE00', '#000000', '#00CC00', '#0000CC']
    cmap_light = ListedColormap(color_list_light[0:numClasses])
    cmap_bold  = ListedColormap(color_list_bold[0:numClasses])
    
    h = 0.03
    k = 0.5
    x_plot_adjust = 0.1
    y_plot_adjust = 0.1
    plot_symbol_size = 50
    
    x_min = svm.X_train_svc_cancer[:, 0].min()
    x_max = svm.X_train_svc_cancer[:, 0].max()
    y_min = svm.X_train_svc_cancer[:, 1].min()
    y_max = svm.X_train_svc_cancer[:, 1].max()
    
    x2, y2 = numpy.meshgrid(numpy.arange(x_min-k, x_max+k, h), numpy.arange(y_min-k, y_max+k, h))

    script_dir = os.path.dirname(__file__) # current file dir
    file = os.path.join(script_dir, '../pickles/svc_cancer_model.sav')
    svc_cancer_model = pickle.load(open(file, "rb"))

    P = svc_cancer_model.predict(numpy.c_[x2.ravel(), y2.ravel()])
    P = P.reshape(x2.shape)
    
    f = plt.figure()
    ax = f.add_subplot(111)
    
    plt.contourf(x2, y2, P, cmap=cmap_light, alpha = 0.8)

    if show_test == "on" and show_train == "on":
        plt.scatter(svm.X_train_svc_cancer[:, 0], svm.X_train_svc_cancer[:, 1], c=svm.y_train_svc_cancer, cmap=cmap_bold, s=plot_symbol_size, edgecolor = 'black')
        plt.xlim(x_min - x_plot_adjust, x_max + x_plot_adjust)
        plt.ylim(y_min - y_plot_adjust, y_max + y_plot_adjust)
        plt.scatter(svm.X_test_svc_cancer[:, 0], svm.X_test_svc_cancer[:, 1], c=svm.y_test_svc_cancer, cmap=cmap_bold, s=plot_symbol_size, marker='^', edgecolor = 'black')

    elif show_test == "on" and show_train == "off":
        plt.xlim(x_min - x_plot_adjust, x_max + x_plot_adjust)
        plt.ylim(y_min - y_plot_adjust, y_max + y_plot_adjust)
        plt.scatter(svm.X_test_svc_cancer[:, 0], svm.X_test_svc_cancer[:, 1], c=svm.y_test_svc_cancer, cmap=cmap_bold, s=plot_symbol_size, marker='^', edgecolor = 'black')

    elif show_test == "off" and show_train == "on":
        plt.scatter(svm.X_train_svc_cancer[:, 0], svm.X_train_svc_cancer[:, 1], c=svm.y_train_svc_cancer, cmap=cmap_bold, s=plot_symbol_size, edgecolor = 'black')
        plt.xlim(x_min - x_plot_adjust, x_max + x_plot_adjust)
        plt.ylim(y_min - y_plot_adjust, y_max + y_plot_adjust)
    
    legend_handles = []
    for i in range(0, len(cancer.cancer_type)):
        patch = mpatches.Patch(color=color_list_bold[i], label=cancer.cancer_type[i])
        legend_handles.append(patch)
    plt.legend(loc=4, handles=legend_handles, handlelength=1, prop={'size': 8})
    
    # set y axis labels to right side.
    ax.yaxis.tick_right()
    
    fig = plt.gcf()
    #convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri =  urllib.parse.quote(string)
    cancer_svc_scatter = uri


    # logistic scatter for cancer
    numClasses = numpy.amax(logistic.y_train_logistic_cancer) + 1
    color_list_light = ['#FFFFAA', '#EFEFEF', '#AAFFAA', '#AAAAFF']
    color_list_bold = ['#EEEE00', '#000000', '#00CC00', '#0000CC']
    cmap_light = ListedColormap(color_list_light[0:numClasses])
    cmap_bold  = ListedColormap(color_list_bold[0:numClasses])
    
    h = 0.03
    k = 0.5
    x_plot_adjust = 0.1
    y_plot_adjust = 0.1
    plot_symbol_size = 50
    
    x_min = logistic.X_train_logistic_cancer[:, 0].min()
    x_max = logistic.X_train_logistic_cancer[:, 0].max()
    y_min = logistic.X_train_logistic_cancer[:, 1].min()
    y_max = logistic.X_train_logistic_cancer[:, 1].max()
    
    x2, y2 = numpy.meshgrid(numpy.arange(x_min-k, x_max+k, h), numpy.arange(y_min-k, y_max+k, h))

    script_dir = os.path.dirname(__file__) # current file dir
    file = os.path.join(script_dir, '../pickles/logistic_cancer_model.sav')
    logistic_cancer_model = pickle.load(open(file, "rb"))

    P = logistic_cancer_model.predict(numpy.c_[x2.ravel(), y2.ravel()])
    P = P.reshape(x2.shape)
    
    f = plt.figure()
    ax = f.add_subplot(111)
    
    plt.contourf(x2, y2, P, cmap=cmap_light, alpha = 0.8)

    if show_test == "on" and show_train == "on":
        plt.scatter(logistic.X_train_logistic_cancer[:, 0], logistic.X_train_logistic_cancer[:, 1], c=logistic.y_train_logistic_cancer, cmap=cmap_bold, s=plot_symbol_size, edgecolor = 'black')
        plt.xlim(x_min - x_plot_adjust, x_max + x_plot_adjust)
        plt.ylim(y_min - y_plot_adjust, y_max + y_plot_adjust)
        plt.scatter(logistic.X_test_logistic_cancer[:, 0], logistic.X_test_logistic_cancer[:, 1], c=logistic.y_test_logistic_cancer, cmap=cmap_bold, s=plot_symbol_size, marker='^', edgecolor = 'black')

    elif show_test == "on" and show_train == "off" :
        plt.xlim(x_min - x_plot_adjust, x_max + x_plot_adjust)
        plt.ylim(y_min - y_plot_adjust, y_max + y_plot_adjust)
        plt.scatter(logistic.X_test_logistic_cancer[:, 0], logistic.X_test_logistic_cancer[:, 1], c=logistic.y_test_logistic_cancer, cmap=cmap_bold, s=plot_symbol_size, marker='^', edgecolor = 'black')

    elif show_test == "off" and show_train == "on":
        plt.scatter(logistic.X_train_logistic_cancer[:, 0], logistic.X_train_logistic_cancer[:, 1], c=logistic.y_train_logistic_cancer, cmap=cmap_bold, s=plot_symbol_size, edgecolor = 'black')
        plt.xlim(x_min - x_plot_adjust, x_max + x_plot_adjust)
        plt.ylim(y_min - y_plot_adjust, y_max + y_plot_adjust)
    
    legend_handles = []
    for i in range(0, len(cancer.cancer_type)):
        patch = mpatches.Patch(color=color_list_bold[i], label=cancer.cancer_type[i])
        legend_handles.append(patch)
    plt.legend(loc=4, handles=legend_handles, handlelength=1, prop={'size': 8})
    # set y axis labels to right side.
    ax.yaxis.tick_right()
    fig = plt.gcf()
    #convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri =  urllib.parse.quote(string)
    cancer_logistic_scatter = uri


    # linear scatter on friedman
    script_dir = os.path.dirname(__file__) # current file dir
    file = os.path.join(script_dir, '../pickles/linear_friedman_model.sav')
    linear_friedman_model = pickle.load(open(file, "rb"))

    f = plt.figure()
    ax = f.add_subplot(111)
     
    plt.scatter(linear.y_test_linear_friedman , linear.y_predicted_label , edgecolors=(0,0,0))
    sns.regplot(x=[linear.y_actual_label.min(), linear.y_actual_label.max()], 
        y=[linear.y_actual_label.min(), linear.y_actual_label.max()], color='red', truncate=False ,ci=None)
    #plt.plot(linear_friedman_model.coef_[0], '^', color='black')

    plt.xlabel('Measured')
    plt.ylabel('Actual')
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
     
    fig = plt.gcf()
    #convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri =  urllib.parse.quote(string)
    friedman_linear_scatter = uri
