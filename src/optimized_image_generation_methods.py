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

def quiver_show_subset(u, v, x_subspace_start, x_subspace_stop, y_subspace_start, y_subspace_stop):
    
    u_subspace = u[y_subspace_start:y_subspace_stop, x_subspace_start:x_subspace_stop]
    v_subspace = v[y_subspace_start:y_subspace_stop, x_subspace_start:x_subspace_stop]

    u_subspace = u_subspace[::-1]
    v_subspace = v_subspace[::-1]
    
    x_subspace = np.arange(x_subspace_start, x_subspace_stop)
    y_subspace = np.arange(y_subspace_start, y_subspace_stop)
    x_subspace, y_subspace = np.meshgrid(x_subspace, y_subspace)
    
    fig = plt.figure(figsize=(24, 18))
    #img = np.zeros((y_subspace_stop - y_subspace_start, x_subspace_stop - x_subspace_start))
    #plt.imshow(img, origin="upper", cmap="binary")
    plt.quiver(u_subspace, v_subspace, color="r")

##############################################

def quiver_show_subset_with_figure(u, v, 
                                   x_subspace_start, x_subspace_stop, 
                                   y_subspace_start, y_subspace_stop
                                  ):
    u_subspace = u[y_subspace_start:y_subspace_stop, x_subspace_start:x_subspace_stop]
    v_subspace = v[y_subspace_start:y_subspace_stop, x_subspace_start:x_subspace_stop]

    x_subspace = np.arange(x_subspace_start, x_subspace_stop)
    y_subspace = np.arange(y_subspace_start, y_subspace_stop)
    x_subspace, y_subspace = np.meshgrid(x_subspace, y_subspace)
    
    fig = plt.figure(figsize=(24, 18))
    img = np.zeros((y_subspace_stop - y_subspace_start, x_subspace_stop - x_subspace_start))
    plt.imshow(img, origin="upper", cmap="binary")
    plt.quiver(u_subspace, v_subspace, color="r")
