import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
from oBehave.plotting_stuff import dffBlockPlot


def singlecellpca(dff,tme,starttime,nPCs = 1,window=(0,.75)):
    X = dffBlockPlot(dff,tme,starttime,window = window,plotme = False);
    pca = PCA()
    a = pca.fit_transform(X)
    a = a[:,:nPCs].T
    a = a[0]
    return a,pca

def sparsity(image_responses):
    '''
    # image responses should be an array of the trial averaged responses to each image
    # sparseness = 1-(sum of trial averaged responses to images / N)squared / (sum of (squared mean responses / n)) / (1-(1/N))
    # N = number of images
    # from tutorial code
    '''
    N = float(len(image_responses))
    ls = ((1-(1/N) * ((np.power(image_responses.sum(axis=0),2)) / (np.power(image_responses,2).sum(axis=0)))) / (1-(1/N)))
    return ls