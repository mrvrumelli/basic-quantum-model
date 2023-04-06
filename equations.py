import math, cmath
from sklearn.preprocessing import normalize

S = 2*math.pow(10,-3)
L = 635*math.pow(10,-9)
PI = math.pi

def activation_function(prediction:float):
    """
    parameter: prediction output
    """ 
    return math.pow(abs(prediction),2)

def first(L_01:float, x:float):
    return ((2*math.pow(PI,1/2)*S)/\
    ((L*L_01)-(2*1j*PI*(S**2))))**(1/2)*\
    cmath.exp((-2*(PI**2)*(S**2)*(x**2))/(4*(PI**2)*\
    S**4+math.pow(L*L_01,2))+1j*x**2*(PI/L*L_01-\
    (4*(PI**3)*S**4/L*L_01*(4*(PI**2)*(S**4)+\
    math.pow(L*L_01,2))))-1j*PI/4)

def first_conj(L_01:float, x:float):
    return ((2*math.pow(PI,1/2)*S)/\
    ((L*L_01)+(2j*PI*(S**2))))**(1/2)*\
    cmath.exp((-2*(PI**2)*(S**2)*(x**2))/(4*(PI**2)*\
    S**4+math.pow(L*L_01,2))-1j*x**2*(PI/L*L_01-\
    (4*(PI**3)*S**4/L*L_01*(4*(PI**2)*(S**4)+\
    math.pow(L*L_01,2))))+1j*PI/4)

def other(x:float, param:float, L_12:float, beta:float):
    return (1-1j)*cmath.exp((-1*PI*math.pow((x-param),2))/\
        ((2*PI*math.pow(beta,2))+(1j*L*L_12)))/((-2j)+\
            ((L*L_12)/(PI*math.pow(beta,2))))**(1/2)

def other_conj(x:float, param:float, L_12:float, beta:float):
    return (1+1j)*cmath.exp((-1*PI*math.pow((x-param),2))/\
        ((2*PI*math.pow(beta,2))-(1j*L*L_12)))/(2j+\
            ((L*L_12)/(PI*math.pow(beta,2))))**(1/2)

def first_derivative(L_01:float, x:float):
    first_L = -(((-1)**(3/4)*cmath.exp(PI*x**2*((1j*L_01*L/4*PI**2*S**4+\
                (L_01*L)**2)-(2*PI*S**2/4*PI*S**4+(L_01*L)**2)))*PI**(1/4)*S*L*\
                (1+2*1j*x**2*(2*PI*S**2+1j*L_01*L)*((4*L_01*(PI*S)**2*L/(4*PI*\
                S**2+(L_01*L)**2)**2)+(1j*(4*PI**3*S**4-L_01**2*PI*L**2)/(4*\
                PI**2*S**4+(L_01*L)**2)**2))))/(math.sqrt(2)*(2*PI*S**2+1j*L_01\
                *S)**2*cmath.sqrt(S/-2j*PI*S**2+L_01*L)))
    
    first_conj_L = (cmath.exp(-(PI*x**2*(8*PI**3*S**6+1j*(L_01*L)**3+2*L_01*PI*\
                    S**2*L*(2J*S**2+L_01*L)))/((4*PI*S**4+(L_01*L)**2)*(4*PI*\
                    S**2**4+(L_01*L)**2)))*-PI**(1/4)*S*L*(1-2*1j*x**2*(2*PI*\
                    S**2-1j*L_01*L)*((4*L_01*(PI*S)**2*L/(4*PI*S**2+\
                    (L_01*L)**2)**2)+(1j*(-4*PI**3*S**4+L_01**2*PI*L**2)/(4*\
                    PI**2*S**4+(L_01*L)**2)**2))))/(math.sqrt(2)*(2*PI*S**2-1J*\
                    L_01*L)**2*math.sqrt(S/2j*PI*S**2+L_01*L))
    
    return first_L, first_conj_L
    
def other_dervivative(x1, x2, L_, b):
    other_L = -((((1/2)+(1j/2))*cmath.exp(-(PI*(x1-x2)**2)/(2*b*PI+1j*L*L_))*L*\
                (2*PI*(b+x1-x2)*(b-x1+x2)+1j*L*L_))/((2*b**2*PI+1j*L*L_)**2*\
                (-2j+(L*L_/b**2*PI))**(1/2)))
    
    other_B = -(((1-1j)*cmath.exp(-(PI*(x1-x2)**2)/(2*b*PI+1j*L*L_))*\
                ((L*L_)**2-2*b**2*PI*(2*PI*(x1-x2)**2+1j*L_*L)))/(b*(2*b**2*PI+\
                1j*L*L_)**2*(-2j+(L*L_/b**2*PI))**(1/2)))

    other_conj_L = (((1/2)+(1j/2))*cmath.exp(-(PI*(x1-x2)**2)/(2*b*PI-1j*L*\
                    L_))*L*(2*1j*PI*(b+x1-x2)*(b-x1+x2)+L*L_))/((2*b**2*PI-1j*\
                    L*L_)**2*(2j+(L*L_/b**2*PI))**(1/2))
    
    other_conj_B = ((1+1j)*cmath.exp(-(PI*(x1-x2)**2)/(2*b*PI-1j*L*L_))*(\
                    (-L*L_)**2-2*b**2*PI*(2*PI*(x1-x2)**2-1j*L_*L)))/(b*(2*b**2\
                    *PI+1j*L*L_)**2*(2j+(L*L_/b**2*PI))**(1/2))
    
    return other_L, other_B, other_conj_L, other_conj_B
    
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
    for i in range(-100,101):
        a_i.append(i*delta_a_j*math.pow(10,3))  
        y_pred.append(activation_function(calculate_y_pred_for_one_layer(L_01, 
            L_12, centerpos1, i*delta_a_j, beta1)))

    return a_i, y_pred, sum(y_pred)*delta_a_j

def two_layers(L_01:float, L_12:float, L_23:float, centerpos1:list, 
    centerpos2:list, delta_a_j:float, beta1:list, beta2:list):

    a_i=[]
    y_pred = []
    for i in range(-100,101): 
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
        result = normalize([y_pred], norm="max")
    
    return a_i, result[0], sum(y_pred)*delta_a_j