import numpy as np
from matplotlib import pyplot as plt
import math
from global_params  import params

plt.close('all')

###############################################

def gabor_filter_even(x,y,delta,theta,f0x,f0y):
    x_hat=np.cos(theta)*x+np.sin(theta)*y
    y_hat=-np.sin(theta)*x+np.cos(theta)*y
    pi=math.pi
    Gabor_k=1./(2*pi*delta**2)*np.exp(-1./(2*delta**2)*((x_hat-params.x0)**2+(y_hat-params.y0)**2))*np.cos(params.xi0*(f0x*x_hat+f0y*y_hat))
    return Gabor_k

def gabor_filter_odd(x,y,delta,theta,f0x,f0y):
    x_hat=np.cos(theta)*x+np.sin(theta)*y
    y_hat=-np.sin(theta)*x+np.cos(theta)*y
    pi=math.pi
    Gabor_k=1./(2*pi*delta**2)*np.exp(-1./(2*delta**2)*((x_hat-params.x0)**2+(y_hat-params.y0)**2))*np.sin(params.xi0*(f0x*x_hat+f0y*y_hat))
    return Gabor_k

###############################################


def filter_mono(t):
    value=params.mono_wm1*np.exp(-(t-params.mono_mium1())**2/(2*params.mono_sigmam1()**2))+params.mono_wm2*np.exp(-(t-params.mono_mium2())**2/(2*params.mono_sigmam2()**2))
    return value

def filter_bi(t):
    value=params.bi_wm1*np.exp(-(t-params.bi_mium1())**2/(2*params.bi_sigmam1()**2))+params.bi_wm2*np.exp(-(t-params.bi_mium2())**2/(2*params.bi_sigmam2()**2))
    return value
