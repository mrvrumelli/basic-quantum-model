import math, cmath

SIGMA = 2*math.pow(10,-3)
LMBDA = 635*math.pow(10,-9)

def activation_function(prediction:float):
    """
    parameter: prediction output
    """ 
    return math.pow(abs(prediction),2)

def first(L_01:float, x:float):
    return ((2*math.pow(math.pi,1/2)*SIGMA)/\
    ((LMBDA*L_01)-(2*1j*math.pi*(SIGMA**2))))**(1/2)*\
    cmath.exp((-2*(math.pi**2)*(SIGMA**2)*(x**2))/(4*(math.pi**2)*\
    SIGMA**4+math.pow(LMBDA*L_01,2))+1j*x**2*(math.pi/LMBDA*L_01-\
    (4*(math.pi**3)*SIGMA**4/LMBDA*L_01*(4*(math.pi**2)*(SIGMA**4)+\
    math.pow(LMBDA*L_01,2))))-1j*math.pi/4)

def other(x:float, param:float, L_12:float, beta:float):
    return (1-1j)*cmath.exp((-1*math.pi*math.pow((x-param),2))/\
        ((2*math.pi*math.pow(beta,2))+(1j*LMBDA*L_12)))/((-2*1j)+\
            ((LMBDA*L_12)/(math.pi*math.pow(beta,2))))**(1/2)
    
def equation_one_layer(L_01:float, L_12:float, x:float, a_j:float, beta:float):
    g_0 = first(L_01, x)
    g_1 = other(x, a_j, L_12, beta)
    return g_0*g_1

def calculate_y_pred_for_one_layer(L_01:list, L_12:list, centerpos1:list, 
    a_j:float, beta:list):

    result = 0
    for j in range(len(centerpos1)): 
        result = result + equation_one_layer(L_01, L_12,
        centerpos1[j], a_j, beta[j])
    
    return result

def one_layer(L_01:float, L_12:float, centerpos1:list, delta_a_j:float, 
    beta1:list):

    a_i=[]
    y_pred = []
    for i in range(-50,51):
        a_i.append(i*delta_a_j*math.pow(10,3))  
        y_pred.append(activation_function(calculate_y_pred_for_one_layer(L_01, 
            L_12, centerpos1, i*delta_a_j, beta1)))

    return a_i, y_pred, sum(y_pred)*delta_a_j

def two_layers(L_01:float, L_12:float, L_23:float, centerpos1:list, 
    centerpos2:list, delta_a_j:float, beta1:list, beta2:list):

    a_i=[]
    y_pred = []
    for i in range(-50,51): 
        a_i.append(i*delta_a_j*math.pow(10,3))  
        result = 0
        first_layer = []
        for j in range(len(centerpos2)):
            first_layer_result = calculate_y_pred_for_one_layer( 
                L_01, L_12, centerpos1, centerpos2[j], beta1)
            second_layer_result = other(centerpos2[j], i*delta_a_j,  L_23, 
                beta2[j])
            result += first_layer_result * second_layer_result
            first_layer.append(activation_function(first_layer_result))
        y_pred.append(activation_function(result))
    
    return a_i, y_pred, sum(y_pred)*delta_a_j