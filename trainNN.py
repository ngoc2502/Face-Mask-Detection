from inspect import Parameter
from symbol import parameters
import numpy as np
from activations import *
from cnn import*
from flatten import*
from fully_conected import*
from handle_data import *
from loss import *
from normalize import *
from Pooling import*

def train_nnet(X,Y,val_X,val_Y,
            batch,
            learning_rate,epoch):
    '''
    Parameters
    ----------
    X : numpy array, shape (N, d + 1).
    Y : numpy array, 
    '''
    