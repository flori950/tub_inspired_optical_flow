import numpy as np
from matplotlib import pyplot as plt
import math

plt.close('all')


class params:
    # scale factor to compress in time (detect faster motions)
    global scale_factor

    scale_factor = 0.1 #original 0.1

    # paths

    kernel_path = 'kernel'

    event_path = '../slider_far/output.txt'

    # n_events = 1e4

    mono_wm1 = 1.95

    mono_wm2 = 0.23

    bi_wm1 = 0.83

    bi_wm2 = -0.34

    x0 = 0

    y0 = 0

    sigma = 3 #original 3

    spatial_sigma = 25 # anders als das normale sigma # original 25

    xi0 = math.pi * 2 # apperently ther is a name tau 

    f0x = 0.057  # units: cycles/pix ?

    f0y = f0x

    half_kernel_size = 8 #original 11  # kernelsize -11 to 11 # pls change kernel size here


    # bi1_mean and scale_bi's are only hyperparameters
    scale_bi1 = 1/2

    scale_bi2 = 3/4

    # Get a subset of events
    i_offset = 2900000 # original 2000000

    num_events = 25000 # original 80000

    #####

    f0x_new = -0.1012  # related to -90 pix/sec of ground truth optical flow

    # Voting spread of each event
    sigma_xy = 1.  # [pixels]

    sigma_t = 1.0  # [time bins]

    # DAVIS camera pixel resolution
    sensor_width = 280

    sensor_height = 180

    # Select sub-region of the image
    band_height = 300   # 40 original

    band_width = 250 # 80 original

    offset_height = 20  # 20 original

    offset_width = 70   # 70 original

    ########################

    def get_scale_factor():
        return scale_factor

    def mono_mium1():
        return 0.55*scale_factor

    def mono_mium2():
        return 0.55*scale_factor

    def mono_sigmam1():
        return 0.10*scale_factor

    def mono_sigmam2():
        return 0.16*scale_factor

    def bi_mium1():
        return 0.44*scale_factor

    def bi_mium2():
        return 0.63*scale_factor

    def bi_sigmam1():
        return 0.12*scale_factor

    def bi_sigmam2():
        return 0.21*scale_factor

    # bi1_mean and scale_bi's are only hyperparameters

    def bi1_mean():
        return 0.2 * scale_factor

    def bi2_mean():
        return params.bi1_mean() * 2

    def mono_mean():
        return (1 * scale_factor + params.bi1_mean() * np.sqrt(36 + 10 * np.log(params.scale_bi1 / params.scale_bi2))) / 10

    # 3 sigma rule for bi1 and mono

    def bi1_sigma():
        return params.bi1_mean() / 3

    def bi2_sigma():
        return params.bi1_sigma() * (3/2)

    def mono_sigma():
        return params.mono_mean() / 3
