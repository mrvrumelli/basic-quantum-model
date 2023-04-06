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

#forward propagation calculation
a_i, y_pred, energy = eq.two_layers(L_01, L_12, L_23, centerpos1, 
    centerpos2, delta_a_j, beta1, beta2)

# import matplotlib.pyplot
# matplotlib.pyplot.plot(a_i,y_pred)
# matplotlib.pyplot.show()

# Calculate difference between target and predict
y_target = [0.00000001]*201
for i in range(201):
    if i >19 and i<40:
        y_target[i] = 1
y_target = np.array(y_target)

y_target9 = [0.00000001]*201
for i in range(201):
    if i >80:
        y_target9[i] = 1
y_target9 = np.array(y_target9)

def kl_divergence(p, q):
    """
    Calculates the KL divergence of two distributions.
    """
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

kl = kl_divergence(y_pred,y_target)
kl9 = kl_divergence(y_pred,y_target9)
# Print the result
print("KL divergence:", kl)
print("KL divergence:", kl9)
def calculate_error(y_pred, y_target):    
        return np.sum(np.square(y_target - y_pred))/ len(y_target)

print(calculate_error(y_pred, y_target))
print(calculate_error(y_pred, y_target9))

import matplotlib.pyplot
matplotlib.pyplot.plot(a_i,y_pred)
matplotlib.pyplot.show()