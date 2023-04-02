#-----------------------------------------------------------------------
# This code helps to import relative modules e.g packages from another #
# directory                                                            #
import sys                                                             #
from os import path                                                    # 
sys.path.append(path.dirname(path.dirname( path.abspath(__file__))))   #
#----------------------------------------------------------------------#

from datasets import fruit, cancer
from algorithms import linear

from datasets.cancer import *
from datasets.fruit import *
from algorithms.linear import *

import io
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
#-----------------------------------------------------------------

fruit_heatmap = None
cancer_heatmap = None
friedman_heatmap = None

def heatmap():
    global fruit_heatmap, cancer_heatmap, friedman_heatmap
    
    # Fruit
    f = plt.figure()   
    ax = f.add_subplot(111)
    sns.heatmap(fruit.heat_fruit.corr(), annot=True, linewidth=.5,)
    fig = plt.gcf()
    #convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri =  urllib.parse.quote(string)
    fruit_heatmap = uri

    # Cancer
    f = plt.figure()
    ax = f.add_subplot(111)
    sns.heatmap(cancer.heat_cancer.corr(), annot=True, linewidth=.5,)
    fig = plt.gcf()
    #convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri =  urllib.parse.quote(string)
    cancer_heatmap = uri

    # Friedman
    f = plt.figure()
    ax = f.add_subplot(111)
    sns.heatmap(linear.heat_friedman.corr(), annot=True, linewidth=.5)
    fig = plt.gcf()
    #convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri =  urllib.parse.quote(string)
    friedman_heatmap = uri