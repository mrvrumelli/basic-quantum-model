import sympy as sp
from sklearn.preprocessing import normalize

S = 2 * 10**(-3)
L = 635 * 10**(-9)
PI = sp.pi

x, x_next, beta, L_01, L_12 = sp.symbols("x x_next beta L_01 L_12", real=True)

def activation_function(prediction:float):
    """
    x_nexteter: prediction output
    """ 
    return abs(prediction)**2

def first(L_01: float, x: float, conjugate: bool = False) -> complex:
    # Numerator and denominator for the root term calculation
    numerator = 2 * sp.sqrt(PI) * S
    denominator = (L * L_01) - (2 * 1j * PI * (S**2))

    # Exponential term calculation
    exponential_term = sp.exp(
        (-2 * (PI**2) * (S**2) * (x**2)) / (4 * (PI**2) * S**4 
        + (L * L_01)**2)+ 1j * x**2 * (PI / L * L_01 
        - (4 * (PI**3) * S**4 / L * L_01 * (4 * (PI**2) * (S**4) 
        + (L * L_01)**2)))- 1j * PI / 4)
    
    # Update deminator and exponential_term with their conjugate
    if conjugate:
        denominator = (L * L_01) + (2 * 1j * PI * (S**2))
        exponential_term = sp.exp(
            (-2 * (PI**2) * (S**2) * (x**2)) / (4 * (PI**2) * S**4 
            + (L * L_01)**2)- 1j * x**2 * (PI / L * L_01 
            - (4 * (PI**3) * S**4 / L * L_01 * (4 * (PI**2) * (S**4) 
            + (L * L_01)**2)))+ 1j * PI / 4)

    # Calculate the root term
    root_term = sp.sqrt(numerator / denominator)

    # Return value of calculation
    return (root_term * exponential_term).evalf()

def first_derivative(L_01_value: float, x_value: float, 
                     conjugate: bool = False) -> complex:
    # Calculate equation from "first" function and take derivative wrt L
    first_derivative_L = sp.diff(first(L_01, x, conjugate), L_01)

    # Assign values to derivative equation result  
    first_derivative_value = first_derivative_L.subs({L_01: L_01_value,
                                                       x: x_value})

    # Return value of derivative
    return first_derivative_value.evalf()

def other(x:float, x_next:float, L_12:float, beta:float, 
          conjugate:bool = False) -> complex:
    numerator = (1 - 1j) * sp.exp((-1 * PI * (x - x_next)**2) 
                                  / ((2 * PI * beta**2) + (1j * L * L_12)))
    denominator = (-2j) + ((L * L_12) / (PI * beta**2))
    
    if conjugate:
        numerator = (1 + 1j) * sp.exp((-1 * PI * (x - x_next)**2) 
                                      / ((2 * PI * beta**2) - (1j * L * L_12)))
        denominator = 2j + ((L * L_12) / (PI * beta**2))
    
    return sp.sqrt(numerator / denominator).evalf()

def other_derivative(x_value:float, x_next_value:float, L_12_value:float, 
                     beta_value:float, conjugate:bool = False, 
                     derivative_L:bool = True) -> complex:
    
    if derivative_L:
        other_derivative_L = sp.diff(other(x, x_next, L_12, beta, 
                                           conjugate), L_12)
        other_derivative_value = other_derivative_L.subs({x: x_value, 
                                                          x_next: x_next_value, 
                                                          L_12: L_12_value, 
                                                          beta: beta_value})
    else:
        other_derivative_B = sp.diff(other(x, x_next, L_12, beta, 
                                           conjugate), beta)
        other_derivative_value = other_derivative_B.subs({x: x_value, 
                                                          x_next: x_next_value, 
                                                          L_12: L_12_value, 
                                                          beta: beta_value})
      
    return other_derivative_value.evalf()

def equation_one_layer(L_01:float, L_12:float, x:float, a_j:float, beta:float):
    g_0 = first(L_01, x)
    g_1 = other(x, a_j, L_12, beta)

    return g_0 * g_1

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
        a_i.append(i * delta_a_j * 10**3)  
        y_pred.append(activation_function(calculate_y_pred_for_one_layer(L_01, 
            L_12, centerpos1, i*delta_a_j, beta1)))

    return a_i, y_pred, sum(y_pred)*delta_a_j

def two_layers(L_01:float, L_12:float, L_23:float, centerpos1:list, 
    centerpos2:list, delta_a_j:float, beta1:list, beta2:list):

    a_i=[]
    y_pred = []
    for i in range(-100,101): 
        a_i.append(i * delta_a_j * 10**3)  
        result = 0
        first_layer = []
        for j in range(len(centerpos2)):
            first_layer_result = calculate_y_pred_for_one_layer( 
                L_01, L_12, centerpos1, centerpos2[j], beta1)
            second_layer_result = other(centerpos2[j], i * delta_a_j,  L_23, 
                beta2[j])
            result += first_layer_result * second_layer_result
            first_layer.append(activation_function(first_layer_result))

        y_pred.append(activation_function(result))
        result = normalize([y_pred], norm="max")
    
    return a_i, result[0], sum(y_pred)*delta_a_j