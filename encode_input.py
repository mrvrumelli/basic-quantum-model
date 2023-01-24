import math, cmath
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from decimal import Decimal

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

axis_color = 'lightgoldenrodyellow'

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, 
    num='Generated Wave Distibution')
fig.subplots_adjust(left=0.5, bottom=0.35)

L_01:float = 100*math.pow(10,-2)
L_12:float = 100*math.pow(10,-2)
L_23:float = 100*math.pow(10,-2)
centerpos1:list = [100*math.pow(10,-6), 100*math.pow(10,-6), 
100*math.pow(10,-6)]
centerpos2:list = [100*math.pow(10,-6), 100*math.pow(10,-6), 
100*math.pow(10,-6)]
delta_a_j:float = math.pow(10,-3)
beta1:list = [10*math.pow(10,-6), 10*math.pow(10,-6), 10*math.pow(10,-6)]
beta2:list = [10*math.pow(10,-6), 10*math.pow(10,-6), 10*math.pow(10,-6)]

# Draw initial plots
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

a1, y1, e1 = one_layer( L_01, L_12, centerpos1, delta_a_j, beta1)
[line1] = ax1.plot(a1, y1, color='deepskyblue')
ax1.set_xlabel('Second layer input positions (mm)')
ax1.set_ylabel('First layer output normalized')
ax1.set_ylim(0, 6)
text1 = ax1.text(0.05, 0.95, r'Total Energy=%.2E' % (Decimal(e1), ), 
transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props)
a2, y2, e2 = two_layers( L_01, L_12,L_23, centerpos1, centerpos2, 
    delta_a_j, beta1, beta2)
[line2] = ax2.plot(a2, y2, color='steelblue')
ax2.set_xlabel('Measurement positions (mm)')
ax2.set_ylabel('Second layer output positions')
ax2.set_ylim(0, 0.03)
text2 = ax2.text(1.25, 0.95, r'Total Energy=%.2E' % (Decimal(e2), ), 
transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props)

# Create sliders for distance 
L01_ax  = fig.add_axes([0.1, 0.2, 0.65, 0.03])
L12_ax  = fig.add_axes([0.1, 0.15, 0.65, 0.03])
L23_ax  = fig.add_axes([0.1, 0.1, 0.65, 0.03])

L01_slider:Slider = Slider(L01_ax, 'L\u2080\u2081(m)', 0.01, 5, valstep=0.01, 
    valinit=L_01)
L12_slider:Slider = Slider(L12_ax, 'L\u2081\u2082(m)', 0.01, 5, valstep=0.01, 
    valinit=L_12)
L23_slider:Slider = Slider(L23_ax, 'L\u2082\u2083(m)', 0.01, 5, valstep=0.01, 
    valinit=L_23)

# Create sliders for first layer center values  
centerpos11_ax  = fig.add_axes([0.1, 0.8, 0.1, 0.03])
centerpos12_ax  = fig.add_axes([0.1, 0.75, 0.1, 0.03])
centerpos13_ax  = fig.add_axes([0.1, 0.7, 0.1, 0.03])

centerpos11_slider:Slider = Slider(centerpos11_ax,
    'Center position\u2081\u2081(\u03BCm)', -5000, 5000, valstep=1,
    valinit=centerpos1[0])
centerpos12_slider:Slider = Slider(centerpos12_ax,
    'Center position\u2081\u2082(\u03BCm)', -5000, 5000, valstep=1,
    valinit=centerpos1[1])
centerpos13_slider:Slider = Slider(centerpos13_ax,
    'Center position\u2081\u2083(\u03BCm)', -5000, 5000, valstep=1,
    valinit=centerpos1[2])

# Create sliders for second layer center values  
centerpos21_ax  = fig.add_axes([0.1, 0.6, 0.1, 0.03]) 
centerpos22_ax  = fig.add_axes([0.1, 0.55, 0.1, 0.03])
centerpos23_ax  = fig.add_axes([0.1, 0.5, 0.1, 0.03])

centerpos21_slider:Slider = Slider(centerpos21_ax,
    'Center position\u2082\u2081(\u03BCm)', -5000, 5000, valstep=1,
    valinit=centerpos2[0])
centerpos22_slider:Slider = Slider(centerpos22_ax,
    'Center position\u2082\u2082(\u03BCm)', -5000, 5000, valstep=1,
    valinit=centerpos2[1])
centerpos23_slider:Slider = Slider(centerpos23_ax,
    'Center position\u2082\u2083(\u03BCm)', -5000, 5000, valstep=1,
    valinit=centerpos2[2])
    
# Create sliders for first layer slit width
b11_ax  = fig.add_axes([0.3, 0.8, 0.1, 0.03]) 
b12_ax  = fig.add_axes([0.3, 0.75, 0.1, 0.03])
b13_ax  = fig.add_axes([0.3, 0.7, 0.1, 0.03])

b11_slider:Slider = Slider(b11_ax, '\u03B2\u2081\u2081(\u03BCm)', 1, 150, 
    valstep=1, valinit=beta1[0])
b12_slider:Slider = Slider(b12_ax, '\u03B2\u2081\u2082(\u03BCm)', 1, 150, 
    valstep=1, valinit=beta1[1])
b13_slider :Slider= Slider(b13_ax, '\u03B2\u2081\u2083(\u03BCm)', 1, 150, 
    valstep=1, valinit=beta1[2])

# Create sliders for first layer slit width
b21_ax  = fig.add_axes([0.3, 0.6, 0.1, 0.03]) 
b22_ax  = fig.add_axes([0.3, 0.55, 0.1, 0.03])
b23_ax  = fig.add_axes([0.3, 0.5, 0.1, 0.03])

b21_slider:Slider = Slider(b21_ax, '\u03B2\u2082\u2081(\u03BCm)', 1, 150, 
    valstep=1, valinit=beta2[0])
b22_slider:Slider = Slider(b22_ax, '\u03B2\u2082\u2082(\u03BCm)', 1, 150, 
    valstep=1, valinit=beta2[1])
b23_slider:Slider = Slider(b23_ax, '\u03B2\u2082\u2083(\u03BCm)', 1, 150,
    valstep=1, valinit=beta2[2])

def sliders_on_changed(val):
    #Conditions for beta range
    if centerpos12_slider.val < centerpos11_slider.val + (3*b11_slider.val):
        centerpos12_slider.val = centerpos11_slider.val + (3*b11_slider.val)
        centerpos12_slider.set_val(centerpos11_slider.val + (3*b11_slider.val))

    if centerpos13_slider.val < centerpos12_slider.val + (3*b12_slider.val):
        centerpos13_slider.val = centerpos12_slider.val + (3*b12_slider.val) 
        centerpos13_slider.set_val(centerpos12_slider.val + (3*b12_slider.val))
    
    if centerpos22_slider.val < centerpos21_slider.val + (3*b21_slider.val):
        centerpos22_slider.val = centerpos21_slider.val + (3*b21_slider.val)
        centerpos22_slider.set_val(centerpos21_slider.val + (3*b21_slider.val))

    if centerpos23_slider.val < centerpos22_slider.val + (3*b22_slider.val):
        centerpos23_slider.val = centerpos22_slider.val + (3*b22_slider.val) 
        centerpos23_slider.set_val(centerpos22_slider.val + (3*b22_slider.val))

    centerpos1 = [centerpos11_slider.val* math.pow(10,-6), 
        centerpos12_slider.val* math.pow(10,-6), 
        centerpos13_slider.val* math.pow(10,-6)]
    centerpos2 = [centerpos21_slider.val* math.pow(10,-6), 
        centerpos22_slider.val* math.pow(10,-6), 
        centerpos23_slider.val* math.pow(10,-6)]
    beta1 = [b11_slider.val* math.pow(10,-6), b12_slider.val* math.pow(10,-6), 
        b13_slider.val* math.pow(10,-6)]
    beta2 = [b21_slider.val* math.pow(10,-6), b22_slider.val* math.pow(10,-6), 
        b23_slider.val* math.pow(10,-6)]

    a1, y1, e1 = one_layer( L01_slider.val, L12_slider.val, centerpos1, 
        delta_a_j, beta1)
    text1.set_text(r'Total Energy=%.2E' % (Decimal(e1), ))
    # Normalization of first plot output
    max1 = max(y1)
    y1[:] = [x / max1 for x in y1]
    line1.set_ydata(y1)
    ax1.set_ylim(0, 1)
    a2, y2, e2 = two_layers( L01_slider.val, L12_slider.val, 
        L23_slider.val, centerpos1, centerpos2, delta_a_j, beta1, beta2)
    text2.set_text(r'Total Energy=%.2E' % (Decimal(e2), ))
    # Normalization of second plot output
    max2 = max(y2)
    y2[:] = [x / max2 for x in y2]

    line2.set_ydata(y2)
    fig.canvas.draw_idle()
    ax2.set_ylim(0, 1)

L01_slider.on_changed(sliders_on_changed)
L12_slider.on_changed(sliders_on_changed)
L23_slider.on_changed(sliders_on_changed)
b11_slider.on_changed(sliders_on_changed)
b12_slider.on_changed(sliders_on_changed)
b13_slider.on_changed(sliders_on_changed)
b21_slider.on_changed(sliders_on_changed)
b22_slider.on_changed(sliders_on_changed)
b23_slider.on_changed(sliders_on_changed)
centerpos11_slider.on_changed(sliders_on_changed)
centerpos12_slider.on_changed(sliders_on_changed)
centerpos13_slider.on_changed(sliders_on_changed)
centerpos21_slider.on_changed(sliders_on_changed)
centerpos22_slider.on_changed(sliders_on_changed)
centerpos23_slider.on_changed(sliders_on_changed)

reset_button_ax = fig.add_axes([0.85, 0.15, 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', color=axis_color, 
    hovercolor='0.975')
def reset_button_on_clicked(mouse_event):
    L01_slider.reset()
    L12_slider.reset()
    L23_slider.reset()
    b11_slider.reset()
    b12_slider.reset()
    b13_slider.reset()
    b21_slider.reset()
    b22_slider.reset()
    b23_slider.reset()
    centerpos11_slider.reset()
    centerpos12_slider.reset()
    centerpos13_slider.reset()
    centerpos21_slider.reset()
    centerpos22_slider.reset()
    centerpos23_slider.reset()

reset_button.on_clicked(reset_button_on_clicked)