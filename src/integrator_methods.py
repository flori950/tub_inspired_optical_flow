#import cv2
import numpy as np
from matplotlib import pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from util import Timer, Event, normalize_image, animate, load_events, plot_3d, event_slice
from filter_methods import gabor_filter_even, gabor_filter_odd, filter_mono, filter_bi
from global_params import params
# from time import time #
from scipy import fftpack, signal

plt.close('all')

###############################################


def integrator(event_data, delta,theta):
    print('Integrating, please wait...')
    events, height, width = event_data.event_list, event_data.height, event_data.width
    with Timer('Integrating  (simple)'):
        image_state = np.zeros((height, width), dtype=np.float32)
        # image_list = []
        for  e in events:
            image_state[e.y, e.x] = image_state[e.y, e.x] +\
                                    e.p*(filter_bi(e.t)*gabor_filter_even(e.x,e.y,delta,theta,params.f0x,params.f0y)+
                                    filter_mono(e.t)*gabor_filter_odd(e.x,e.y,delta,theta,params.f0x,params.f0y))
    plt.imshow(image_state, cmap=cm.coolwarm)
    return image_state

###############################################

def conv_integrator(image, kernal_size):
    print('Convolution, please wait...')
    kerDimx = kernal_size
    kerDimy = kernal_size
    kernel = np.ones((kerDimy, kerDimx), dtype=np.float32)
    print("kernel", kernel)
    with Timer('Convolution (simple)'):
        result = signal.convolve(image, kernel, mode="same")
    return result