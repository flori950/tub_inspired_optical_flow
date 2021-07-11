import numpy as np
from matplotlib import pyplot as plt
import math
from global_params import params

plt.close('all')

###############################################


def gabor_filter_even(x, y, delta, theta, f0x, f0y):
    x_hat = np.cos(theta)*x+np.sin(theta)*y
    y_hat = -np.sin(theta)*x+np.cos(theta)*y
    pi = math.pi
    return 1./(2*pi*delta**2)*np.exp(-1./(2*delta**2)*((x_hat-params.x0)
                                                          ** 2+(y_hat-params.y0)**2))*np.cos(params.xi0*(f0x*x_hat+f0y*y_hat))
   

def gabor_filter_odd(x, y, delta, theta, f0x, f0y):
    x_hat = np.cos(theta)*x+np.sin(theta)*y
    y_hat = -np.sin(theta)*x+np.cos(theta)*y
    pi = math.pi
    return 1./(2*pi*delta**2)*np.exp(-1./(2*delta**2)*((x_hat-params.x0)
                                                          ** 2+(y_hat-params.y0)**2))*np.sin(params.xi0*(f0x*x_hat+f0y*y_hat))

###############################################


def filter_mono(t):
    return params.mono_wm1*np.exp(-(t-params.mono_mium1())**2/(2*params.mono_sigmam1(
    )**2))+params.mono_wm2*np.exp(-(t-params.mono_mium2())**2/(2*params.mono_sigmam2()**2))



def filter_bi(t):
    return params.bi_wm1*np.exp(-(t-params.bi_mium1())**2/(2*params.bi_sigmam1(
    )**2))+params.bi_wm2*np.exp(-(t-params.bi_mium2())**2/(2*params.bi_sigmam2()**2))
    

###############################################


def temporal_filter(t, mu, sigma):
    temp_filter = np.exp(-(t - mu)**2 / (2 * sigma**2))
    return temp_filter


###############################################


def spatial_gabor_filter_even(x, y, sigma, theta, f0x, f0y):
    x_hat = np.cos(theta) * x + np.sin(theta) * y
    y_hat = -np.sin(theta) * x + np.cos(theta) * y

    gabor_first = np.exp(-1 * ((x_hat - f0x)**2 + (y_hat - f0y)**2) * (2 * math.pi**2) / sigma**2)
    gabor_second = np.cos(params.xi0 * (f0x * x_hat + f0y * y_hat))
    return (params.xi0 / sigma**2) * gabor_first * gabor_second

    


def spatial_gabor_filter_odd(x, y, sigma, theta, f0x, f0y):
    x_hat = np.cos(theta) * x + np.sin(theta) * y
    y_hat = -np.sin(theta) * x + np.cos(theta) * y

    gabor_first = np.exp(-1 * ((x_hat - f0x)**2 + (y_hat - f0y)**2) * (2 * math.pi**2 / sigma**2))
    gabor_second = np.sin(params.xi0 * (f0x * x_hat + f0y * y_hat))
    return (params.xi0 / sigma**2) * gabor_first * gabor_second

   