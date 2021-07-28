import numpy as np
from matplotlib import pyplot as plt
import math
import cv2
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


###############################################

def normalize(u, v):
    beta_response = 1
    alpha_p = 0.1
    alpha_q = 0.002
    sigma_response = 3.6

    center = math.ceil(sigma_response * 3)
    size = center * 2 + 1

    filter_gaussian = cv2.getGaussianKernel(size, sigma_response)

    def relu(x):
        return x * (x > 0)

    uv_response = np.sqrt(u**2 + v**2)

    relu_response = relu(uv_response)
    gaussian_response = cv2.filter2D(relu_response, -1, filter_gaussian)
    normalized_response = beta_response * uv_response / (alpha_p + uv_response + relu(gaussian_response / alpha_q))

    ratio = normalized_response / uv_response
    u_normalized = ratio * u
    v_normalized = ratio * v

    return u_normalized, v_normalized

###############################################

def filter_bi_spacial(t_spatial):  
    return -1 * params.scale_bi1 * temporal_filter(t_spatial, params.bi1_mean(), params.bi1_sigma())+ params.scale_bi2 * temporal_filter(t_spatial, params.bi2_mean(), params.bi2_sigma())

def filter_mono_spacial(t_spatial):
    return temporal_filter(t_spatial, params.mono_mean(), params.mono_sigma())

###############################################

def temporal_filter(t, mu, sigma):
    temp_filter = np.exp(-(t - mu)**2 / (2 * sigma**2))
    return temp_filter


###############################################

def filter_mono(t):
    return params.mono_wm1*np.exp(-(t-params.mono_mium1())**2/(2*params.mono_sigmam1(
    )**2))+params.mono_wm2*np.exp(-(t-params.mono_mium2())**2/(2*params.mono_sigmam2()**2))



def filter_bi(t):
    return params.bi_wm1*np.exp(-(t-params.bi_mium1())**2/(2*params.bi_sigmam1(
    )**2))+params.bi_wm2*np.exp(-(t-params.bi_mium2())**2/(2*params.bi_sigmam2()**2))
    
###############################################

def gaussian(t, mu, sigma):
        return np.exp(-0.5 * ((t-mu)/sigma)**2)