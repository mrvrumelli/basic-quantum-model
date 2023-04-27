# image size = 3x3
# Each layer has 9x2  = 18 slits

import math
import random
import equations as eq
import numpy as np
import pandas as pd

MICRO = math.pow(10,-6)
TOTALPIXEL = 9 # CHANGE
SLITNUM = 4 # The number of slits each pixel has

def create_df_for_image(image:list):
    total_slit_number = list(range(1,(TOTALPIXEL*SLITNUM)+1))
    beta = [10] * len(total_slit_number)
    centerpos = list(range(10,(sum(beta)*3)+10,30))
    pixel_no = total_slit_number[:len(total_slit_number)//4]*SLITNUM
    pixel_value = [0, 0, 1, 1]

    layer_no = [1] * len(total_slit_number)
    random.shuffle(pixel_no)
    first_layer_data = {'LayerIndex' : layer_no,
            'SlitNo': total_slit_number,
            'CenterPosition': centerpos,
            'Beta': beta,
            'PixelNo': pixel_no
            }

    first_df = pd.DataFrame(first_layer_data).sort_values(by = ['PixelNo'])
    first_df['PixelValue'] = pixel_value * int(len(total_slit_number) / 4)
    first_df = first_df.sort_values(by = ['SlitNo'])
    first_df['CenterPosition'] *= MICRO
    first_df['Beta'] *= MICRO

    layer_no = [2] * len(total_slit_number)
    random.shuffle(pixel_no)
    second_layer_data = {'LayerIndex' : layer_no,
            'SlitNo': total_slit_number,
            'CenterPosition': centerpos,
            'Beta': beta,
            'PixelNo': pixel_no
            }

    second_df = pd.DataFrame(second_layer_data).sort_values(by = ['PixelNo'])
    second_df['PixelValue'] = pixel_value * int(len(total_slit_number) / 4)
    second_df = second_df.sort_values(by = ['SlitNo'])
    second_df['CenterPosition'] *= MICRO
    second_df['Beta'] *= MICRO

    image = np.array(image).flatten() # CHANGE
    image_data = {'PixelNo' : list(range(1,len(image)+1)),
            'PixelValue': image.tolist()
            }
    image_df = pd.DataFrame(image_data)

    first_df = pd.merge(first_df, image_df, on=['PixelNo', 'PixelValue'], 
    how='inner').sort_values(by=['SlitNo'], ignore_index=True)
    second_df = pd.merge(second_df, image_df, on=['PixelNo', 'PixelValue'], 
    how='inner').sort_values(by=['SlitNo'], ignore_index=True)

    result_df = pd.concat([first_df, second_df])

    return result_df

example_image = [[1, 1, 0],[0, 1, 0],[1, 1, 1]]
result_df = create_df_for_image(example_image)

L_01:float = 100*math.pow(10,-2)
L_12:float = 100*math.pow(10,-2)
L_23:float = 100*math.pow(10,-2)
centerpos1 = result_df.loc[result_df['LayerIndex'] == 1,'CenterPosition']
centerpos2 = result_df.loc[result_df['LayerIndex'] == 2,'CenterPosition']
delta_a_j:float = math.pow(10,-5)
beta1 = result_df.loc[result_df['LayerIndex'] == 1,'Beta']
beta2 = result_df.loc[result_df['LayerIndex'] == 1,'Beta']

def kl_divergence(p, q):
    """
    Calculates the KL divergence of two distributions.
    """
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

#kl = kl_divergence(y_pred,y_target)
#kl9 = kl_divergence(y_pred,y_target9)
# Print the result
#print("KL divergence:", kl)
#print("KL divergence:", kl9)
def calculate_error(y_pred, y_target):    
        return np.sum(np.square(y_target - y_pred))/ len(y_target)

#print(calculate_error(y_pred, y_target))

def test(L_01, L_12, L_23, centerpos1:list, 
        centerpos2:list, delta_a_j, beta1, beta2):
    #forward propagation calculation
    a_i, y_pred, energy = eq.forward_prop(centerpos1, centerpos2, delta_a_j, L_01, L_12, L_23, beta1, beta2)

    # Calculate difference between target and predict
    y_target = [0.00000001]*201
    for i in range(201):
        if i >19 and i<40:
            y_target[i] = 1
    y_target = np.array(y_target)

    loss = calculate_error(y_pred, y_target)  
    d_loss_B1 = 2*np.sum((y_target - y_pred)* eq.derivative_for_L_12_and_B_1(centerpos1,centerpos2, L_01,L_12,beta1,False))/len(y_target)
    d_loss_B2 = 2*np.sum((y_target - y_pred)* eq.derivative_for_L_23_and_B_2(centerpos1,centerpos2,delta_a_j, L_01,L_12,L_23,beta1,beta2,False))/len(y_target)
    d_loss_L01 = 2*np.sum((y_target - y_pred)* eq.derivative_for_L_01(centerpos1,L_01))/len(y_target)
    d_loss_L12 = 2*np.sum((y_target - y_pred)* eq.derivative_for_L_12_and_B_1(centerpos1,centerpos2, L_01,L_12,beta1,True))/len(y_target)
    d_loss_L23 = 2*np.sum((y_target - y_pred)* eq.derivative_for_L_23_and_B_2(centerpos1,centerpos2,delta_a_j, L_01,L_12,L_23,beta1,beta2,True))/len(y_target)
    
    # d_loss variables represent equations with derivative of parameters
    p_derivatives_B1 = [d_loss_B1]
    p_derivatives_L01 = [d_loss_L01]
    p_derivatives_B2 = [d_loss_B2]
    p_derivatives_L12 = [d_loss_L12]
    p_derivatives_L23 = [d_loss_L23]

    return  p_derivatives_B1, p_derivatives_L01, p_derivatives_B2, p_derivatives_L12, p_derivatives_L23, loss

def optimize(L_01, L_12, L_23, centerpos1, centerpos2, delta_a_j, beta1, beta2, learning_rate):
    epoch = 0
    error = 999

    errors = list()
    epochs = list()

    while (epoch <= 1000) and (error > 9e-4):
        
        loss_ = 0
        p_derivatives = test(L_01, L_12, L_23, centerpos1, centerpos2, delta_a_j, beta1, beta2)           
        
        L_01 = L_01 - (learning_rate * np.array(p_derivatives[1]))
        L_12 = L_12 - (learning_rate * np.array(p_derivatives[3]))
        L_23 = L_23 - (learning_rate * np.array(p_derivatives[4]))
        beta1 = beta1 - (learning_rate * np.array(p_derivatives[0]))
        beta2 = beta2 - (learning_rate * np.array(p_derivatives[2]))

            # Evaluate the results
        #for index, feature_value_test in enumerate(TOTALPIXEL):
        loss = (learning_rate * np.array(p_derivatives[5]))
        loss_ += loss

        errors.append(loss_/TOTALPIXEL)
        epochs.append(epoch)
        error = errors[-1]
        epoch += 1

        print('Epoch {}. loss: {}'.format(epoch, errors[-1]))

    
    return errors


print(optimize(L_01, L_12, L_23, centerpos1, centerpos2, delta_a_j, beta1, beta2, 0.1))