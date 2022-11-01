from array import array
from asyncio import streams
from distutils.command.build_scripts import first_line_re
from importlib.util import LazyLoader
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

def predict(x: list,x_next: list, s, l, L, beta, a_i: float):
    """
    x: input array
    s: sigma value of quation
    l: lamda value of equation
    L: distance between layers
    a_i: last layer last item
    complex # i = j in python (1j)
    """
    global first_layer, middle_layer, last_layer
       
    a = -(2*math.pow(math.pi*s*x[1],2))/(4*math.pow(math.pi,2)*math.pow(s,4)+math.pow(l*L,2))
    first_layer = x[0]*np.exp(a+1j*(math.pi/(l*L)+(2*math.pi*math.pow(s,2)*a)/(l*L)))

    middle_layer = (1-1j*math.exp((-math.pi*math.pow(x-x_next,2))/(2*math.pi*beta**2 +1j*l*L)))/math.sqrt(-2j+(l*L/math.pi*beta**2 ))

    last_layer = (1-1j*math.exp((-math.pi*math.pow(x-a_i,2))/(2*math.pi*beta**2 +1j*l*L)))/math.sqrt(-2j+(l*L/math.pi*beta**2 ))
    
    prediction = first_layer * middle_layer * last_layer
    prediction = activation_function(prediction)
    return prediction

def forward_propagation(x,x_next, s, l, L: list, beta, a_i, y):

    y_pred = predict(x,x_next, s, l, L, beta, a_i)
    loss = (y_pred - y)**2   
    d_loss = 2*(y_pred - y)
    
    return y_pred, loss, d_loss

def backpropagation(d_loss, x , x_next,sigma_0, l, L:list, beta, a_i):
    partial_derivatives_B = list()
    partial_derivatives_L = list()

    for layer_index in range(len(x)):
        param1 = 4*math.pi**2*sigma_0**4
        param2 = l*L[layer_index]
        param3 = 2*math.pi*sigma_0**2*x[layer_index]**2
        param6 = math.pi*beta**2        
        param5 = math.sqrt(-2j + (param2/param6)) 

        if layer_index == 0:#first layer            
            equation_for_L = x[layer_index]*(2*param2*param3*l/(param1+param2**2)**2+1j(2*math.pi*sigma_0**2*param3*(param1+3*param2**2)/param2*L[layer_index]*(param1+param2**2)**2-math.pi/(param2*L[layer_index])))*math.exp(-param3/(param1+param2**2)+1j(math.pi/param2-2*math.pi*sigma_0**2*param3/param2*(param1+param2**2)))

            partial_derivatives_B.append('no result')
            partial_derivatives_L.append(d_loss*equation_for_L*middle_layer*last_layer)
        
        else:
            if layer_index == len(x):
                param4 = math.pi*(x[layer_index]-a_i)**2/(2*param6+1j*param2)
            else:
                param4 = math.pi*(x[layer_index]-x[last_layer+1])**2/(2*param6+1j*param2)
            
            equation_for_B = -2*math.pi*beta*(1-1j)*math.exp(-param4)*param4*param5-param2/(param6)**2
            equation_for_L = (1-1j)*math.exp(-param4)*((-1j*l*param4/(2*param6-1j*param2)**2)*(2j+param2/param6)-(l/2*param6*(2j+param2/param6)**(1./3.)))

            if layer_index == len(x):
                partial_derivatives_B.append(d_loss*equation_for_B*first_layer*middle_layer)
                partial_derivatives_L.append(d_loss*equation_for_L*first_layer*middle_layer)
            else:
                ml_before = (1-1j*math.exp((-math.pi*math.pow(x[:layer_index]-x_next[:layer_index],2))/(2*math.pi*beta**2 +1j*l*L)))/math.sqrt(-2j+(l*L/math.pi*beta**2 ))
                ml_after = (1-1j*math.exp((-math.pi*math.pow(x[layer_index+1:]-x_next[layer_index+1:],2))/(2*math.pi*beta**2 +1j*l*L)))/math.sqrt(-2j+(l*L/math.pi*beta**2 ))
                partial_derivatives_B.append(d_loss*equation_for_B*first_layer*middle_layer*ml_before*ml_after)
                partial_derivatives_L.append(d_loss*equation_for_L*first_layer*middle_layer*ml_before*ml_after)        

    return partial_derivatives_B, partial_derivatives_L