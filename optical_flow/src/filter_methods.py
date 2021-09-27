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

# ###############################################

# def get_axis_indices(x, start_val, end_val, filter_apothem, filter_size):
#     # @param x: index for the selected axis
#     # @param start_val: is 0 for whole image
#     # @param end_val: is band_width / height for whole image
#     # @param filter_dilation: on x axis / y axis etc.
#     # @return x_start, x_stop: the image region to be summed with filter output.
#     # @return x_filter_start, x_filter_stop: the filter region to be added to image region
    
#     x_start_region = x - filter_apothem
        
#     if x_start_region < start_val:
#         x_start = start_val
#         x_filter_start = -x_start_region
#     else:
#         x_start = x_start_region
#         x_filter_start = 0
    
#     x_stop_region = x + filter_apothem + 1
    
#     if x_stop_region > end_val:
#         x_stop = end_val
#         x_filter_stop = filter_size - (x_stop_region - end_val)
#     else:
#         x_stop = x_stop_region
#         x_filter_stop = filter_size
        
#     return x_start, x_stop, x_filter_start, x_filter_stop

# ###############################################

# def get_filtered_image(event_data, t_end,
#                        start_x_img, start_y_img, stop_x_img, stop_y_img, 
#                        filter_amount, filter_apothem, f0x,
#                        temporal_mono_filter, temporal_bi1_filter, temporal_bi2_filter,
#                        scale_biphasic1, scale_biphasic2,
#                        even_filters, odd_filters
#                       ):
    
#     even_filters = np.asarray(even_filters)
#     even_filters = even_filters.reshape(even_filters.shape[1], even_filters.shape[2], -1)
    
#     odd_filters = np.asarray(odd_filters)
#     odd_filters = odd_filters.reshape(odd_filters.shape[1], odd_filters.shape[2], -1)
    
    
#     pixels_x = stop_x_img - start_x_img
#     pixels_y = stop_y_img - start_y_img
    
#     # order is reversed for band_height - width for x, y indexing as in a picture
#     grid_vox = np.zeros((pixels_y, pixels_x, filter_amount), dtype=np.float64)
    
#     stop_index = len(event_data)
    
#     filter_size = 2 * filter_apothem + 1
    
#     for index in np.arange(0, stop_index):
#         t, x, y = event_subset[index]

#         # Compute temporal filter

#         t_diff = t_end - t

#         temporal_monophasic = temporal_mono_filter.get(t_diff)
#         temporal_biphasic = scale_biphasic1 * temporal_bi1_filter.get(t_diff)
#         temporal_biphasic += scale_biphasic2 * temporal_bi2_filter.get(t_diff)

#         x_start, x_stop, x_filter_start, x_filter_stop = \
#             get_axis_indices(x, 0, band_width, filter_apothem, filter_size)
#         y_start, y_stop, y_filter_start, y_filter_stop = \
#             get_axis_indices(y, 0, band_height, filter_apothem, filter_size)

#         even_filter_val = temporal_biphasic * even_filters
#         odd_filter_val = temporal_monophasic * odd_filters        
#         filter_val = even_filter_val + odd_filter_val

#         grid_vox[y_start:y_stop, x_start:x_stop] += \
#             filter_val[y_filter_start:y_filter_stop, x_filter_start:x_filter_stop]
    
#     return grid_vox