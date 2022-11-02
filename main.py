from array import array
from asyncio import streams
from distutils.command.build_scripts import first_line_re
from xmlrpc.client import Boolean
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#First layer doesn't have beta value, it is assumed that beta list begin with value that represent there is no value in first index(like - or x)
def activation_function(prediction):
    """
    parameter: prediction output
    """ 
    return pow(abs(np.exp(prediction)),2)

def predict(x: list, s, l, L: list, beta: list, a_i: float):
    """
    x: input array
    s: sigma value of quation
    l: lamda value of equation
    L: distance array between layers
    a_i: last layer last item
    complex # i = j in python (1j)
    """
    global first_layer, other_layers, x_next

    x_next = x[2:].append(a_i)

    a = -(2*math.pow(math.pi*s*x[1],2))/(4*math.pow(math.pi,2)*math.pow(s,4)+math.pow(l*L[0],2))
    first_layer = x[0]*np.exp(a+1j*(math.pi/(l*L[0])+(2*math.pi*math.pow(s,2)*a)/(l*L[0])))
    prediction = first_layer

    for layer_index in range(1,len(x)):
        try:
            other_layers = (1-1j*math.exp((-math.pi*math.pow(x[layer_index]-x_next[layer_index],2))/(2*math.pi*beta[layer_index]**2 +1j*l*L[layer_index])))/math.sqrt(-2j+(l*L[layer_index]/math.pi*beta[layer_index]**2 ))
        except ValueError:
            print("input length doesn't match with input next!!")
        
        prediction = prediction * other_layers

    prediction = activation_function(prediction)
    return prediction

def forward_propagation(x: list, s, l, L: list, beta: list, a_i, y):

    y_pred = predict(x, s, l, L, beta, a_i)
    loss = (y_pred - y)**2   
    d_loss = 2*(y_pred - y)
    
    return y_pred, loss, d_loss

def calculate_other_layers(before_flag: Boolean, after_flag: Boolean, x: list, layer_index: int, beta: list, L: list, l):
    #check if the calculated layer is second or the last one
    before, after = 1
    if before_flag:
        for index in range(1,layer_index):
            before = (1-1j*math.exp((-math.pi*math.pow(x[index]-x_next[index],2))/(2*math.pi*beta[index]**2 +1j*l*L[index])))/math.sqrt(-2j+(l*L[index]/math.pi*beta[index]**2 ))

    if after_flag:
        for index in range(layer_index+1, len(x)):
            after = (1-1j*math.exp((-math.pi*math.pow(x[index]-x_next[index],2))/(2*math.pi*beta[index]**2 +1j*l*L[index])))/math.sqrt(-2j+(l*L[index]/math.pi*beta[index]**2 ))

    return before * after

def backpropagation(d_loss, x: list, sigma_0, l, L:list, beta:list):
    partial_derivatives_B = list()
    partial_derivatives_L = list()

    for layer_index in range(len(x)):
        param1 = 4*math.pi**2*sigma_0**4
        param2 = l*L[layer_index]
        param3 = 2*math.pi*sigma_0**2*x[layer_index]**2
        param6 = math.pi*beta[layer_index]**2   
        param4 = math.pi*(x[layer_index]-x_next)**2/(2*param6+1j*param2)     
        param5 = math.sqrt(-2j + (param2/param6)) 

        if layer_index == 0:#first layer            
            equation_for_L = x[layer_index]*(2*param2*param3*l/(param1+param2**2)**2+1j(2*math.pi*sigma_0**2*param3*(param1+3*param2**2)/param2*L[layer_index]*(param1+param2**2)**2-math.pi/(param2*L[layer_index])))*math.exp(-param3/(param1+param2**2)+1j(math.pi/param2-2*math.pi*sigma_0**2*param3/param2*(param1+param2**2)))

            partial_derivatives_B.append('no result')
            partial_derivatives_L.append(d_loss*equation_for_L*other_layers)
        
        else:     
            equation_for_B = -2*math.pi*beta[layer_index]*(1-1j)*math.exp(-param4)*param4*param5-param2/(param6)**2
            equation_for_L = (1-1j)*math.exp(-param4)*((-1j*l*param4/(2*param6-1j*param2)**2)*(2j+param2/param6)-(l/2*param6*(2j+param2/param6)**(1./3.)))

            layer_value = 1
            if layer_index == 1:
                layer_value = calculate_other_layers(False, True, x,layer_index, beta, L, l)
            elif layer_index == len(x):
                layer_value = calculate_other_layers(True, False, x,layer_index, beta, L, l)
            else:
                layer_value = calculate_other_layers(True, True, x,layer_index, beta, L, l)

            partial_derivatives_B.append(d_loss*equation_for_B*first_layer*layer_value)
            partial_derivatives_L.append(d_loss*equation_for_L*first_layer*layer_value)        

    return partial_derivatives_B, partial_derivatives_L