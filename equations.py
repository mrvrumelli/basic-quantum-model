import math, cmath
import numpy as np
from sklearn.preprocessing import normalize

S = 2 * 10**(-3) # sigma value
Lambda_ = 635 * 10**(-9) # lambda value
PI = math.pi # pi value


## _equation_values: foton gelirken parametrelere gore aldigi degerler
##_values: o layerin alabilecegi her durum icin kosullari calistirip ilgili sonucu dondurur
def activation_function(prediction:float):
    """
    parameter: prediction output
    """ 
    return abs(prediction)**2

def first_layer_equation_values(centerpos1: float, L_01: float, conjugate: bool = False) -> complex:
    # Returns value between input and first layer (g0)
    """
    x: Slit position as input
    L_01: Distance between input and first layer
    conjugate: Check equation is conjugate form or not
    """
    # Numerator and denominator for the root term calculation
    numerator = 2 * math.sqrt(PI) * S
    denominator = (Lambda_ * L_01) - (2 * 1j * PI * (S**2))

    # Exponential term calculation
    exponential_term = cmath.exp(
        (-2 * (PI**2) * (S**2) * (centerpos1**2)) / (4 * (PI**2) * S**4 
        + (Lambda_ * L_01)**2)+ 1j * centerpos1**2 * (PI / Lambda_ * L_01 
        - (4 * (PI**3) * S**4 / Lambda_ * L_01 * (4 * (PI**2) * (S**4) 
        + (Lambda_ * L_01)**2)))- 1j * PI / 4)
    
    # Update deminator and exponential_term with their conjugate
    if conjugate:
        denominator = (Lambda_ * L_01) + (2 * 1j * PI * (S**2))
        exponential_term = cmath.exp(
            (-2 * (PI**2) * (S**2) * (centerpos1**2)) / (4 * (PI**2) * S**4 
            + (Lambda_ *
             L_01)**2)- 1j * centerpos1**2 * (PI / Lambda_ * L_01 
            - (4 * (PI**3) * S**4 / Lambda_ * L_01 * (4 * (PI**2) * (S**4) 
            + (Lambda_ * L_01)**2)))+ 1j * PI / 4)

    # Calculate the root term
    root_term = (numerator / denominator)**(1/2)

    # Return value of calculation
    return root_term * exponential_term

def other_layers_equation_values(centerpos1:float, centerpos2:float, L_value:float, beta:float, 
          conjugate:bool = False) -> complex:
    # Returns value between first layer and second layer (g2) or,
    # Returns value between second layer and last layer (g3)
    """
    x: Slit position as input
    centerpos2: Slit position on next layer 
    L_value: Distance between two layers
    beta: Slit width
    conjugate: Check equation is conjugate form or not
    """
    if type(centerpos1) == list or type(centerpos2) == list:
         a = 5
    numerator = (1 - 1j) * cmath.exp((-1 * PI * (centerpos1 - centerpos2)**2) 
                                  / ((2 * PI * beta**2) + (1j * Lambda_ * L_value)))
    denominator = (-2j) + ((Lambda_ * L_value) / (PI * beta**2))
    
    if conjugate:
        numerator = (1 + 1j) * cmath.exp((-1 * PI * (centerpos1 - centerpos2)**2) 
                                      / ((2 * PI * beta**2) - (1j * Lambda_ * L_value)))
        denominator = 2j + ((Lambda_ * L_value) / (PI * beta**2))
    
    return (numerator / denominator)**(1/2)

def first_layer_derivative_value(centerpos1: float, L_01_value: float,  
                     conjugate: bool = False) -> complex:
    # Returns derivative of first layer (g0)
    """
    x: Slit position as input
    L_01: Distance between input and first layer
    conjugate: Check equation is conjugate form or not
    """
    if conjugate:
        term1 = (PI**(-1/4)) * Lambda_ * S
        term2 = 1 - 2j * centerpos1**2 * (2 * PI * S**2 - 1j * Lambda_ * L_01_value)
        term3 = (4 * PI**2 * Lambda_ * L_01_value * S**2) / (Lambda_**2 * L_01_value**2 + 4 
                * PI * S**4)**2+1j * (PI * Lambda_**2 * L_01_value**2 - 4 * PI**3 * 
                S**4)/(Lambda_**2 * L_01_value**2 + 4 * PI**2 * S**4)**2
        numerator = term1 * term2 * term3
        exp_term = cmath.exp((-PI * centerpos1**2)*(1j * Lambda_**3 * L_01_value**3
                        + 2 * PI * Lambda_ * L_01_value * S**2 * (Lambda_ * L_01_value + 2j 
                        * S**2) + 8 * PI**3 * S**6)/(Lambda_**2 * L_01_value**2 + 4 * 
                        PI * S**4) * (Lambda_**2 * L_01_value**2 + 4 * PI**2 * S**4))
        
        denominator = (cmath.sqrt(2) * (2 * PI * S**2 - 1j * Lambda_ * 
                        L_01_value)**2)*(cmath.sqrt(S / (Lambda_ * L_01_value + 2j 
                        * PI * S**2)))

        result = numerator * exp_term / denominator
    else:
        numerator = (-1)**(3/4) * (PI**(1/4)) * Lambda_ * S * (1 + 2j * centerpos1**2 * 
                        (2 * PI * S**2 + 1j * Lambda_ * L_01_value) * 
                        ((4 * PI**2 * Lambda_ * L_01_value * S**2) / 
                        (Lambda_**2 * L_01_value**2 + 4 * PI * S**4)**2 + 
                        (1j * (4 * PI**3 * S**4 - PI * Lambda_**2 * L_01_value**2)) / 
                        (Lambda_**2 * L_01_value**2 + 4 * PI**2 * S**4)**2))
        denominator = math.sqrt(2) * (2 * PI * S**2 + 1j * Lambda_ * L_01_value)**2\
                        * cmath.sqrt(S / (Lambda_ * L_01_value - 2j * PI * S**2))
        result = -1 * (numerator / denominator)
    return result

def other_layers_derivative_value(centerpos1:float, centerpos2:float, L_value:float, 
                     beta:float, conjugate:bool = False, 
                     derivative_L:bool = True) -> complex:   
        # Returns derivative of second layer (g2) or,
        # Returns derivative of last layer (g3)
        """
        x: Slit position as input
        centerpos2: Slit position on next layer 
        L_value: Distance between two layers
        beta: Slit width
        conjugate: Check equation is conjugate form or not
        derivative_L: Check which parameter is calculated (L or beta)
        """ 
        if derivative_L and conjugate:
                term1 = (1/2 + 1j/2) * Lambda_
                exp_term = cmath.exp(-math.pi * (centerpos1-centerpos2)**2 / (2 * math.pi * beta**2 - 1j * Lambda_ * L_value))
                term2 = (Lambda_ * L_value + 2j * math.pi * (beta + centerpos1 - centerpos2) * (beta - centerpos1 + centerpos2))
                denominator = ((2 * math.pi * beta**2 - 1j * Lambda_ * L_value)**2 *
                        cmath.sqrt(Lambda_ * L_value / (math.pi * beta**2) + 2j))
                return (term1 * exp_term * term2) / denominator        
        elif derivative_L and not conjugate:
                term = (-(1/2 + 1j/2) * Lambda_)*(2 * math.pi * (beta + centerpos1 - centerpos2) * (beta - centerpos1 + centerpos2) + 1j * Lambda_ * L_value)
                exp_term = cmath.exp(-math.pi * (centerpos1-centerpos2)**2 / (2 * math.pi * beta**2 + 1j * Lambda_ * L_value))
                denominator = ((2 * math.pi * beta**2 + 1j * Lambda_ * L_value)**2 *
                        cmath.sqrt(Lambda_ * L_value / (math.pi * beta**2) - 2j))
                return (term * exp_term) / denominator
        
        elif conjugate and not derivative_L: 
                term = (1 + 1j) * cmath.exp(-math.pi * (centerpos1-centerpos2)**2 / (2 * math.pi * beta**2 - 1j * Lambda_ * L_value))
                num = (-Lambda_**2 * L_value**2 + 2 * math.pi * beta**2 * (2 * math.pi * (centerpos1-centerpos2)**2 - 1j * Lambda_ * L_value))
                denominator = (beta * (2 * math.pi * beta**2 - 1j * Lambda_ * L_value)**2 *
                        cmath.sqrt(Lambda_ * L_value / (math.pi * beta**2) + 2j))
                return (term * num) / denominator
        else:
                term = (1 - 1j) * cmath.exp(-math.pi * (centerpos1-centerpos2)**2 / (2 * math.pi * beta**2 + 1j * Lambda_ * L_value))
                num = (Lambda_**2 * L_value**2 - 2 * math.pi * beta**2 * (2 * math.pi * (centerpos2-centerpos2)**2 + 1j * Lambda_ * L_value))
                denominator = (beta * (2 * math.pi * beta**2 + 1j * Lambda_ * L_value)**2 *
                        cmath.sqrt(Lambda_ * L_value / (math.pi * beta**2) - 2j))
                return (-term * num) / denominator

def first_layer_values(centerpos1:float, L_01: float, conjugate:bool = False, derivative:bool = False) -> complex:
    # Find equation result of first layer propagation as forward or backward   
    if derivative:
        g_0 = first_layer_derivative_value(centerpos1, L_01, conjugate)       
    else: 
        g_0 = first_layer_equation_values(centerpos1, L_01, conjugate)
    
    return g_0

def other_layer_values(centerpos1:float, centerpos2:float, L_value:float, beta:float, conjugate:bool = False, derivative:bool = False, derivative_L:bool = True):
    # Find equation result of second or last layer propagation as forward or backward   
    if derivative:
        g_ = other_layers_derivative_value(centerpos1, centerpos2, L_value, beta, conjugate, derivative_L)       
    else: 
        g_ = other_layers_equation_values(centerpos1, centerpos2, L_value, beta, conjugate)
    
    return g_

def sum_second(centerpos1:list, centerpos2:float, L_01:float, L_12:float, beta:list, conjugate:bool = False, derivative:bool = False, derivative_L:bool = True):
    # Calculates the sum of the result of each slit in the first layer for a slit in the second layer.
    ##iki slit sonuclarini tek tek topluyor. bunu hem backprop hem de forward icin guncelle
    result = 0
    for i in range(len(centerpos1)): 
        result +=  first_layer_values(centerpos1[i], L_01, conjugate) *\
                   other_layer_values(centerpos1[i], centerpos2, L_12, beta[i], conjugate, derivative, derivative_L)
            
    return result

def forward_prop(centerpos1:list, centerpos2:list, delta_a_j:float, L_01:float, L_12:float, L_23:float, beta1:list, beta2:list, conjugate:bool = False, derivative:bool = False, derivative_L:bool = True):
    a_i=[]
    y_pred = []
    for i in range(-100,101): 
        a_i.append(i * delta_a_j * 10**3)  
        result = 0
        for j in range(len(centerpos2)):# calculate ypred for each slits in second layer
            first_layer_result = sum_second(centerpos1, centerpos2[j], L_01, L_12, beta1, conjugate)
            second_layer_result = other_layer_values(centerpos2[j], i * delta_a_j,  L_23, 
                beta2[j], conjugate, derivative, derivative_L)
            result += first_layer_result * second_layer_result
           
        y_pred.append(activation_function(result))
        # if len(y_pred) >2:
        #     a=5
        #result = normalize([y_pred], norm="max")
    
    #return a_i, result[0], sum(y_pred)*delta_a_j
    return a_i, y_pred, sum(y_pred)*delta_a_j

def derivative_for_L_01(centerpos:list, L_01):
    d_L_01 = 0
    for i in range(len(centerpos)):
        g_0 = first_layer_equation_values(centerpos[i], L_01) # g_0
        g_0_conj = first_layer_equation_values(centerpos[i], L_01, True) # g_0*
        d_g_0 = first_layer_derivative_value(centerpos[i], L_01) # g_0'    
        d_g_0_conj = first_layer_derivative_value(centerpos[i], L_01, True) # g_0*'

        d_L_01 += d_g_0 * g_0_conj + g_0 * d_g_0_conj
    return d_L_01

def derivative_for_L_12_and_B_1(centerpos1:list, centerpos2:list, L_01:float, L_12:float, beta:list, L:bool = True):
    # False ise ikinci layerdaki beta degeri icin turev aliniyor, true ise L_12 icin turev alinmis oluyor
    ## BURADA BIRINDE CENTERPOS1 LIST BIRINDE FLOAT ALMISIM 
    result = 0
    for i in range(len(centerpos2)): 
        d_sum_g_1 = sum_second(centerpos1, centerpos2[i], L_01, L_12, beta, False, True, L)
        g_1_conj = sum_second(centerpos1, centerpos2[i], L_01, L_12, beta, True, False)
        d_sum_g_1_conj = sum_second(centerpos1, centerpos2[i], L_01, L_12, beta, True, True, L)
        g_1 = sum_second(centerpos1, centerpos2[i], L_01, L_12, beta)

        result += d_sum_g_1 * g_1_conj + d_sum_g_1_conj * g_1
    return result

def derivative_for_L_23_and_B_2(centerpos1, centerpos2, delta_a_j, L_01, L_12, L_23, beta1, beta2, L:bool = True):
    d_sum_g_2 = forward_prop(centerpos1, centerpos2, delta_a_j, L_01, L_12, L_23, beta1, beta2, False, True, L)[1]
    g_2_conj = forward_prop(centerpos1, centerpos2, delta_a_j, L_01, L_12, L_23, beta1, beta2, True, False)[1]
    d_sum_g_2_conj = forward_prop(centerpos1, centerpos2, delta_a_j, L_01, L_12, L_23, beta1, beta2, True, True, L)[1]
    g_2 = forward_prop(centerpos1, centerpos2, delta_a_j, L_01, L_12, L_23, beta1, beta2)[1]
    
    result = d_sum_g_2[1] * g_2_conj[1] + d_sum_g_2_conj[1] * g_2[1]
    return result