from array import array
from asyncio import streams
from distutils.command.build_scripts import first_line_re
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def activation_function(prediction):
    """
    parameter: prediction output
    """ 
    return pow(abs(np.exp(prediction)),2)

def predict(x: array,x_next: array, s, l, L, beta, a_i: float):
    """
    x: input array
    s: sigma value of quation
    l: lamda value of equation
    L: distance between layers
    a_i: last layer last item
    complex # i = j in python (1j)
    """
    a = -(2*math.pow(math.pi*s*x[1],2))/(4*math.pow(math.pi,2)*math.pow(s,4)+math.pow(l*L,2))
    first_layer = x[0]*np.exp(a+1j*(math.pi/(l*L)+(2*math.pi*math.pow(s,2)*a)/(l*L)))

    middle_layer = (1-1j*math.exp((-math.pi*math.pow(x-x_next,2))/(2*math.pi*beta**2+1j*l*L)))/math.sqrt(-2j+(l*L/math.pi*beta**2))

    last_layer = (1-1j*math.exp((-math.pi*math.pow(x-a_i,2))/(2*math.pi*beta**2+1j*l*L)))/math.sqrt(-2j+(l*L/math.pi*beta**2))
    
    prediction = first_layer * middle_layer * last_layer
    prediction = activation_function(prediction)
    return prediction

def forward_propagation(x,x_next, s, l, L, beta, a_i, layer, y):

    y_pred = predict(x,x_next, s, l, L, beta, a_i, layer)
    loss = (y_pred - y)**2   
    d_loss = 2*(y_pred - y)
    
    return y_pred, loss, d_loss
def backpropagation(d_loss, layer):
    partial_derivatives_B = list()
    partial_derivatives_L = list()

    return 0