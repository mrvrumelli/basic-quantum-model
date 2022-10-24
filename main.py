from array import array
from asyncio import streams
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

def equations_of_neurons(x: array,s,l,L):
    """
    x: input array
    s: sigma value of quation
    l: lamda value of equation
    L: distance between layers
    """
    a = -(2*math.pow(math.pi*s*x[1],2))/(4*math.pow(math.pi,2)*math.pow(s,4)+math.pow(l*L,2))
    first_layer = x[0]*np.exp(a+1j*(math.pi/(l*L)+(2*math.pi*math.pow(s,2)*a)/(l*L)))
    middle_layer = (1-1j*math.exp())
    # last_layer =

def predict(s, x, X, H, h, bias):
    """
    s: scale value
    x: slit position
    X: Transpose of x
    H: Quadratic weight
    h: Weight
    complex # i = j in python (1j)
    """
    return prediction 




    #prediction = np.dot(pow(s,2)* np.dot(X, H), x) + s * np.dot(X, h) + bias
    #3. sayfadaki buyuk formul
    prediction = activation_function(prediction)
    
    return prediction