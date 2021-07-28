import os
#import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from util import Timer, Event, normalize_image, animate, load_events, plot_3d, event_slice
from filter_methods import filter_bi_spacial, filter_mono_spacial, gabor_filter_even, gabor_filter_odd, filter_mono, filter_bi, temporal_filter, spatial_gabor_filter_even, spatial_gabor_filter_odd, normalize
from optimized_image_generation_methods import quiver_show_subset
from integrator_methods import integrator, conv_integrator
from global_params import params
# from time import time #
from scipy import fftpack, signal

plt.close('all')

with Timer('Loading'):
    # param.sn_events = 1e4

    path_to_events = params.event_path  # change path if folder is different
    print("Loads events from", path_to_events)
    event_data = load_events(path_to_events)  # load_events from utils file
print("_______________________")

###############################################
# plot_3d from utils file

# with Timer("Plotting..."):
#     plot_3d(event_data, len(event_data.event_list))
# print("_______________________")

###############################################
print("Nyquist sampling frequency calculation:")
dt_res = 0.03*params.get_scale_factor()
ft_nyquist = 1. / dt_res  # Nyquist sampling frequency
print(ft_nyquist, "Hz")

###############################################

t_spatial = np.arange(0., 0.7 * params.get_scale_factor(), dt_res)

Tmono_spatial = filter_mono_spacial(t_spatial)
Tbi_spatial = filter_bi_spacial(t_spatial)

#image genereation spatial

fig2, ax1 = plt.subplots()
val_max = np.max([np.max(np.abs(Tmono_spatial)), np.max(np.abs(Tbi_spatial))])
ax1.plot(t_spatial, Tmono_spatial / val_max, "g-")
ax1.plot(t_spatial, Tbi_spatial / val_max, "b-")
ax1.plot(t_spatial, (Tmono_spatial+Tbi_spatial) / val_max, "r-")
ax1.set_title("temporal filters (mono- and bi-phasic)-spatial")
ax1.set_xlabel("time")
plt.savefig("../output_figures/temporal_filters_mono_bi_spatial.png")
plt.grid()
plt.show()


###########################################################


t = np.arange(0., 1.2*params.get_scale_factor(), dt_res)
Tmono = filter_mono(t)
Tbi = filter_bi(t)

# image genereation

fig2, ax1 = plt.subplots()
ax1.plot(t, Tmono/np.max(np.abs(Tmono)), 'g-')
ax1.plot(t, Tbi/np.max(np.abs(Tbi)), 'b-')
ax1.set_title("temporal filters (mono- and bi-phasic)- original")
ax1.set_xlabel('time')
plt.savefig("../output_figures/temporal_filters_mono_bi.png")
plt.grid()
plt.show()
print("_______________________")

##############################################

kernel_size = 2*params.half_kernel_size+1
x = np.linspace(-params.half_kernel_size, params.half_kernel_size, kernel_size)
print("Kernel")
print(x)
xv, yv = np.meshgrid(x, x)

dx = 1.  # pixels
# Nyquist sampling frequency in (x,y)-space, in cycles / pix
ft_nyquist = 1. / dx

# So f0x and f0y cannot be larger than 0.5 cycles / pix
# initialicing f0x_new, f0y_new in params
print("_______________________")

##############################################

print("Gabor filter calculation")


G_even = gabor_filter_even(xv, yv, params.sigma, 0., params.f0x_new, 0.)
G_odd = gabor_filter_odd(xv, yv, params.sigma, 0., params.f0x, 0.)

fig = plt.figure()
plt.subplot(1, 2, 1), plt.imshow(G_even)
plt.subplot(1, 2, 2), plt.imshow(G_odd)
plt.savefig("../output_figures/gabor_filters_0.png")


##############################################

G_even_spatial = spatial_gabor_filter_even(xv, yv, params.spatial_sigma, 0, params.f0x, params.f0y)
G_odd_spatial = spatial_gabor_filter_odd(xv, yv, params.spatial_sigma, 0, params.f0x, params.f0y)
fig = plt.figure()
plt.subplot(1, 2, 1), plt.imshow(G_even_spatial)
plt.subplot(1, 2, 2), plt.imshow(G_odd_spatial)
plt.savefig("../output_figures/gabor_filters_spatial_0.png")
# plt.grid()
# plt.show()

##############################################

G_even = gabor_filter_even(xv, yv, params.sigma, np.pi/4, params.f0x_new, 0.)
G_odd = gabor_filter_odd(xv, yv, params.sigma, np.pi/4, params.f0x_new, 0.)
fig = plt.figure()
plt.subplot(1, 2, 1), plt.imshow(G_even)
plt.subplot(1, 2, 2), plt.imshow(G_odd)
plt.savefig("../output_figures/gabor_filters_pi_4.png")


##############################################

G_even_spatial = spatial_gabor_filter_even(xv, yv, params.spatial_sigma, np.pi / 4, params.f0x, params.f0y)
G_odd_spatial = spatial_gabor_filter_odd(xv, yv, params.spatial_sigma, np.pi / 4, params.f0x, params.f0y)
fig = plt.figure()
plt.subplot(1, 2, 1), plt.imshow(G_even_spatial)
plt.subplot(1, 2, 2), plt.imshow(G_odd_spatial)
plt.savefig("../output_figures/gabor_filters_spatial_pi_4.png")
# plt.grid()
# plt.show()

##############################################


G_even = gabor_filter_even(xv, yv, params.sigma, np.pi/2, params.f0x_new, 0.)
G_odd = gabor_filter_odd(xv, yv, params.sigma, np.pi/2, params.f0x_new, 0.)
fig = plt.figure()
plt.subplot(1, 2, 1), plt.imshow(G_even)
plt.subplot(1, 2, 2), plt.imshow(G_odd)
plt.savefig("../output_figures/gabor_filters_pi_2.png")

##############################################

G_even_spatial = spatial_gabor_filter_even(xv, yv, params.spatial_sigma, np.pi / 2, params.f0x, params.f0y)
G_even_spatial = spatial_gabor_filter_odd(xv, yv, params.spatial_sigma, np.pi / 2, params.f0x, params.f0y)
fig = plt.figure()
plt.subplot(1, 2, 1), plt.imshow(G_even_spatial)
plt.subplot(1, 2, 2), plt.imshow(G_odd_spatial)
plt.savefig("../output_figures/gabor_filters_spatial_pi_2.png")
# plt.grid()
# plt.show()

##############################################


G_even = gabor_filter_even(xv, yv, params.sigma, np.pi/4*3, params.f0x_new, 0.)
G_odd = gabor_filter_odd(xv, yv, params.sigma, np.pi/4*3, params.f0x_new, 0.)
fig = plt.figure()
plt.subplot(1, 2, 1), plt.imshow(G_even)
plt.subplot(1, 2, 2), plt.imshow(G_odd)
plt.savefig("../output_figures/gabor_filters_pi_4x3.png")

##############################################

G_even_spatial = spatial_gabor_filter_even(xv, yv, params.spatial_sigma, np.pi / 4 * 3, params.f0x, params.f0y)
G_odd_spatial = spatial_gabor_filter_odd(xv, yv, params.spatial_sigma, np.pi / 4 * 3, params.f0x, params.f0y)
fig = plt.figure()
plt.subplot(1, 2, 1), plt.imshow(G_even_spatial)
plt.subplot(1, 2, 2), plt.imshow(G_odd_spatial)
plt.savefig("../output_figures/gabor_filters_spatial_pi_4x3.png")
# plt.grid()
# plt.show()

print("generated all Gabor in output Folder")
print("_______________________")

###############################################

space_time_kernel_1 = G_even[:, :, None]*Tbi      
space_time_kernel_2 = G_odd[:, :, None]*Tmono 
space_time_kernel_full = space_time_kernel_1 + space_time_kernel_2

print("Saving original Kernel to ", params.kernel_path, " folder")

np.save(params.kernel_path + '/space_time_kernel1.npy',
        space_time_kernel_1)  # adjust folder if needed
np.save(params.kernel_path + '/space_time_kernel2.npy',
        space_time_kernel_2)  # adjust folder if needed
np.save(params.kernel_path + '/space_time_kernel_combined.npy',
        space_time_kernel_full)  # adjust folder if needed

###############################################

space_time_kernel_1_spatial = G_even_spatial[:, :, None]*Tbi_spatial 
space_time_kernel_2_spatial = G_odd_spatial[:, :, None]*Tmono_spatial  #change to _spatial if needed #original G_odd
space_time_kernel_full_spatial = space_time_kernel_1_spatial + space_time_kernel_2_spatial

print("Saving original Kernel to ", params.kernel_path, " folder (spatial)")

np.save(params.kernel_path + '/space_time_kernel1_spatial.npy',
        space_time_kernel_1_spatial)  # adjust folder if needed
np.save(params.kernel_path + '/space_time_kernel2_spatial.npy',
        space_time_kernel_2_spatial)  # adjust folder if needed
np.save(params.kernel_path + '/space_time_kernel_combined_spatial.npy',
        space_time_kernel_full_spatial)  # adjust folder if needed



print("_______________________")

##############################################

# Filter bank
print(" Filter Bank spatial")

angle_spatial = np.arange(0., np.pi, np.pi/8) 
print(angle_spatial, " Angles spatial")

num_orientations_spatial = len(angle_spatial)

filters_spatial = []
for an_spatial in angle_spatial:
    G_even_spatial = spatial_gabor_filter_even(xv, yv, params.spatial_sigma, an_spatial, params.f0x_new, 0.) #sigma_spatial = 25
    G_odd_spatial = spatial_gabor_filter_odd(xv, yv, params.spatial_sigma, an_spatial, params.f0x_new, 0.)
    space_time_kernel_gabor_even_and_odd_spatial = G_even_spatial[:, :, None]*Tbi_spatial + G_odd_spatial[:, :, None]*Tmono_spatial # make a minus instead of a plus changes everything
    filters_spatial.append(space_time_kernel_gabor_even_and_odd_spatial) 

print("Size of each original kernel (spatial)")
# size of each kernel
space_time_kernel_gabor_even_and_odd_spatial.shape          
print(space_time_kernel_gabor_even_and_odd_spatial.shape)   

print("_______________________")


# Filter bank
print(" Filter Bank ")

angle = np.arange(0., np.pi, np.pi/8) 
print(angle, " Angles")

num_orientations = len(angle)


filters = []
for an in angle:
    G_even = gabor_filter_even(xv, yv, params.sigma, an, params.f0x_new, 0.) #sigma = 3
    G_odd = gabor_filter_odd(xv, yv, params.sigma, an, params.f0x_new, 0.)
    #G_even_spatial = spatial_gabor_filter_even(xv, yv, params.spatial_sigma, an, params.f0x_new, 0.) #sigma_spatial = 25
    #G_odd_spatial = spatial_gabor_filter_odd(xv, yv, params.spatial_sigma, an, params.f0x_new, 0.)
    #space_time_kernel_gabor_even_and_odd_spatial = G_even_spatial[:, :, None]*Tbi_spatial + G_odd_spatial[:, :, None]*Tmono_spatial # make a minus instead of a plus changes everything
    space_time_kernel_gabor_even_and_odd = G_even[:, :, None]*Tbi + G_odd[:, :, None]*Tmono
    filters.append(space_time_kernel_gabor_even_and_odd) #change to _spatial if needed # original  space_time_kernel_gabor_even_and_od

print("Size of each original kernel")
# size of each kernel
space_time_kernel_gabor_even_and_odd.shape          #spatial above
print(space_time_kernel_gabor_even_and_odd.shape) 

print("_______________________")

############################################## Error

# fig = plt.figure(figsize=(50, 15))
# # auto_add_to_figure = False
# ax = Axes3D(fig)  # error in newer pyton
# # fig.add_axes(Axes3D())
# # auto_add_to_figure = False

# X = np.arange(-np.pi*5, np.pi*5, np.pi/10)
# Y = np.arange(-np.pi*5, np.pi*5, np.pi/10)
# R = np.arange(0, np.pi, np.pi/8)
# X, Y = np.meshgrid(X, Y)
# for i in range(len(filters)):

#     Z = gabor_filter_even(X, Y, 2.5, R[i], params.f0x_new, params.f0y)
#     ax = fig.add_subplot(2, int(len(filters)/2), i+1, projection='3d')
#     ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                     cmap=cm.coolwarm, linewidth=0, antialiased=False)
#     # ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.25)
#     cset = ax.contour(X, Y, Z, zdir='z', offset=-np.pi/4, cmap=cm.coolwarm)
#     cset = ax.contour(X, Y, Z, zdir='x', offset=-np.pi/4, cmap=cm.coolwarm)
#     cset = ax.contour(X, Y, Z, zdir='y', offset=3*np.pi/4, cmap=cm.coolwarm)
#     # cb = fig.colorbar(p, shrink=0.5)

#     ax.set_title(" filter bank "+str(i)+"/8 pi")
#     ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
#     ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
#     ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
# print("_______________________")

##############################################

print("Space time kernel of 1:")
num_bins_t = space_time_kernel_1.shape[2]
print(num_bins_t)
print("_______________________")

##############################################

print("Space time kernel spatial of 1:")
num_bins_t_spatial = space_time_kernel_1_spatial.shape[2]
print(num_bins_t_spatial)
print("_______________________")

##############################################

# Plot an x-t slice of each component of the space-time filter
# Fig 3 in the 2014 paper

print(" Plot an x-t slice of each component of the space-time filter ")
slice_idx = int(kernel_size/2)  # index of the central y-slice

max = np.max(space_time_kernel_full)
min = np.min(space_time_kernel_full)
fig = plt.figure()
plt.figure(figsize=(12, 12))

plt.subplot(1, 3, 1),
plt.imshow(np.transpose(space_time_kernel_1[:, slice_idx, :]), cmap='jet_r')
# plt.gca().invert_yaxis()
plt.gca().set_aspect((1. * kernel_size / num_bins_t))
plt.xlabel("x pixel"), plt.ylabel(
    "time bin"), plt.title("t-biphasic - space-even")

# plt.grid()
# plt.show()


plt.subplot(1, 3, 2),
plt.imshow(np.transpose(space_time_kernel_2[:, slice_idx, :]), cmap='jet_r')
plt.gca().set_aspect((1. * kernel_size / num_bins_t))
plt.xlabel("x pixel"), plt.ylabel(
    "time bin"), plt.title("t-monophasic - space-odd")

# plt.grid()
# plt.show()


plt.subplot(1, 3, 3),
plt.imshow(np.transpose(
    space_time_kernel_full[:, slice_idx, :]), vmin=min, vmax=max, cmap='jet_r')
plt.gca().set_aspect((1. * kernel_size / num_bins_t))
plt.xlabel("x pixel"), plt.ylabel("time bin"), plt.title("Combined filter")
plt.savefig("../output_figures/combined_filters.png")
# plt.grid()
# plt.show()

print("_______________________")

##############################################
# spatial space time filter


print(" Plot an x-t slice of each component of the space-time filter (spatial)")
slice_idx_spatial = int(kernel_size/2)  # index of the central y-slice

max = np.max(space_time_kernel_full_spatial)
min = np.min(space_time_kernel_full_spatial)
fig = plt.figure()
plt.figure(figsize=(12, 12))

plt.subplot(1, 3, 1),
plt.imshow(np.transpose(space_time_kernel_1_spatial[:, slice_idx_spatial, :]), cmap='jet_r')
# plt.gca().invert_yaxis()
plt.gca().set_aspect((1. * kernel_size / num_bins_t_spatial))
plt.xlabel("x pixel"), plt.ylabel(
    "time bin"), plt.title("t-biphasic - space-even spatial")

# plt.grid()
# plt.show()


plt.subplot(1, 3, 2),
plt.imshow(np.transpose(space_time_kernel_2_spatial[:, slice_idx_spatial, :]), cmap='jet_r')
plt.gca().set_aspect((1. * kernel_size / num_bins_t_spatial))
plt.xlabel("x pixel"), plt.ylabel(
    "time bin"), plt.title("t-monophasic - space-odd spatial")

# plt.grid()
# plt.show()


plt.subplot(1, 3, 3),
plt.imshow(np.transpose(
    space_time_kernel_full_spatial[:, slice_idx, :]), vmin=min, vmax=max, cmap='jet_r')
plt.gca().set_aspect((1. * kernel_size / num_bins_t_spatial))
plt.xlabel("x pixel"), plt.ylabel("time bin"), plt.title("Combined filter spatial")
plt.savefig("../output_figures/combined_filters_spatial.png")
# plt.grid()
# plt.show()

print("_______________________")

##############################################
# Show Filters

print(" Show filters ")
fig = plt.figure()
plt.figure(figsize=(14, 14))
num_filters = len(filters)
# print(filters[0][:,5,:])
for i in range(len(filters)):
    plt.subplot(1, num_filters, i+1),
    plt.imshow(np.transpose(filters[i][:, slice_idx, :]), cmap='jet_r')
    plt.imshow(np.transpose(filters[i][:, 5, :]), cmap='jet_r')
    # plt.gca().invert_yaxis()
    plt.gca().set_aspect((1. * kernel_size / num_bins_t))
    plt.xlabel("x pixel"), plt.ylabel(
        "time bin"), plt.title("space-time filter")
plt.savefig("../output_figures/spacetime_filters.png")
# plt.grid()
# plt.show()

print("_______________________")

##############################################

# Show Filters spatial

print(" Show filters spatial")
fig = plt.figure()
plt.figure(figsize=(14, 14))
num_filters_spatial = len(filters_spatial)
# print(filters[0][:,5,:])
for i in range(len(filters_spatial)):
    plt.subplot(1, num_filters_spatial, i+1),
    plt.imshow(np.transpose(filters_spatial[i][:, slice_idx_spatial, :]), cmap='jet_r')
    plt.imshow(np.transpose(filters_spatial[i][:, 5, :]), cmap='jet_r')
    # plt.gca().invert_yaxis()
    plt.gca().set_aspect((1. * kernel_size / num_bins_t_spatial))
    plt.xlabel("x pixel"), plt.ylabel(
        "time bin"), plt.title("space-time filter spatial")
plt.savefig("../output_figures/spacetime_filters_spatial.png")
# plt.grid()
# plt.show()

print("_______________________")

##############################################

# Compute response of the filter to a given input. See description in LNAI8774 2014
# Bio-inspired optic flow from event-based neuromorphic sensor input, Section 3

ev_subset = []
# print("event_lit=",event_data.event_list)
with Timer("reading data"):
    for i, e in enumerate(event_data.event_list):
        if (i >= params.i_offset and i < params.i_offset+params.num_events):
            # print("event=",e.x,e.y,e.time,e.p)
            ev_subset.append([e.x, e.y, e.t, e.p])

ev_subset = np.array(np.array(ev_subset))
print(ev_subset.shape)
print("_______________________")

##############################################

ev_subset_spatial = []
# print("event_lit=",event_data.event_list)
with Timer("reading data spatial"):
    for i, e in enumerate(event_data.event_list):
        if (i >= params.i_offset and i < params.i_offset+params.num_events):
            # print("event=",e.x,e.y,e.time,e.p)
            ev_subset_spatial.append([e.x, e.y, e.t, e.p])

ev_subset_spatial = np.array(np.array(ev_subset_spatial))
print(ev_subset_spatial.shape)
print("_______________________")

##############################################

# Time span of the events
print("Time span of the events")
t_min = np.min(ev_subset[:, 2])
t_max = np.max(ev_subset[:, 2])
dt_span = t_max - t_min
print(dt_span, " sec")

num_bins_tev = int(np.ceil(dt_span / dt_res))
print(num_bins_tev, " bins in which the events fits")
print("_______________________")

##############################################

# Time span of the events spatial
print("Time span of the events")
t_min_spatial = np.min(ev_subset_spatial[:, 2])
t_max_spatial = np.max(ev_subset_spatial[:, 2])
dt_span_spatial = t_max_spatial - t_min_spatial
print(dt_span_spatial, " sec")

num_bins_tev_spatial = int(np.ceil(dt_span_spatial / dt_res))
print(num_bins_tev_spatial, " bins in which the events fits")
print("_______________________")

##############################################

# First, try the synchronous solution: convert the events to a 3D voxel grid by voting:
# each events fills in some part of the 3D grid, according to a Gaussian weights in space and time

# 3D Meshgrid
print("3D Meshgrid")
x_ = range(0, params.band_width)
y_ = range(0, params.band_height)
z_ = range(0, num_bins_tev)
yv, xv, tv = np.meshgrid(y_, x_, z_, indexing='ij')

grid_vox = np.zeros((params.band_height, params.band_width,
                    num_bins_tev), dtype=np.float32)
# t_begin = time.time()
with Timer("synchronous solution computing..."):
    for ie in range(0, params.num_events):
        # x,y,t coordinates in the voxel grid (could be non-integer)
        x = ev_subset[ie, 0]
        y = ev_subset[ie, 1]
        p = ev_subset[ie, 3]
        if ((params.offset_height <= y) and (y < params.offset_height + params.band_height)
                and (params.offset_width <= x) and (x < params.offset_width + params.band_width)):
            x -= params.offset_width
            y -= params.offset_height
            t = ev_subset[ie, 2]
            t_bin_coord = (t - t_min)/dt_res
            assert t_bin_coord >= 0, "must non-negative"
            assert t_bin_coord <= num_bins_tev, "must be smaller than the number of bins"

            # Brute-force voting: using xv,yv,tv in spite of them having many zeros for each event
            # units: pixel^2 / pixel^2
            exponent_space = -((xv-x)**2 + (yv-y)**2) / \
                (2*(params.sigma_xy**2))
            exponent_time = -((tv-t_bin_coord)**2) / \
                (2*(params.sigma_t**2))      # units: bin^2 / bin^2
            grid_vox += np.exp(exponent_space + exponent_time)

# print (np.round_(t.time() - t_begin, 3), 'sec elapsed')
print(grid_vox.shape)
# print(grid_vox)
# plt.imshow(grid_vox)
print("_______________________")

##############################################


# 3D Meshgrid
print("3D Meshgrid spatial")
x__spatial = range(0, params.band_width)
y__spatial = range(0, params.band_height)
z__spatial = range(0, num_bins_tev_spatial)
yv_spatial, xv_spatial, tv_spatial = np.meshgrid(y__spatial, x__spatial, z__spatial, indexing='ij')

grid_vox_spatial = np.zeros((params.band_height, params.band_width,
                    num_bins_tev_spatial), dtype=np.float32)
# t_begin = time.time()
with Timer("synchronous solution computing..."):
    for ie_spatial in range(0, params.num_events):
        # x,y,t coordinates in the voxel grid (could be non-integer)
        x_spatial = ev_subset_spatial[ie_spatial, 0]
        y_spatial = ev_subset_spatial[ie_spatial, 1]
        p_spatial = ev_subset_spatial[ie_spatial, 3]
        if ((params.offset_height <= y_spatial) and (y_spatial < params.offset_height + params.band_height)
                and (params.offset_width <= x_spatial) and (x_spatial < params.offset_width + params.band_width)):
            x_spatial -= params.offset_width
            y_spatial -= params.offset_height
            t_spatial = ev_subset_spatial[ie, 2]
            t_bin_coord_spatial = (t_spatial - t_min_spatial)/dt_res
            assert t_bin_coord_spatial >= 0, "must non-negative"
            assert t_bin_coord_spatial <= num_bins_tev_spatial, "must be smaller than the number of bins"

            # Brute-force voting: using xv,yv,tv in spite of them having many zeros for each event
            # units: pixel^2 / pixel^2
            exponent_space_spatial = -((xv_spatial-x_spatial)**2 + (yv_spatial-y_spatial)**2) / \
                (2*(params.sigma_xy**2))
            exponent_time_spatial = -((tv-t_bin_coord_spatial)**2) / \
                (2*(params.sigma_t**2))      # units: bin^2 / bin^2
            grid_vox_spatial += np.exp(exponent_space_spatial + exponent_time_spatial)

# print (np.round_(t.time() - t_begin, 3), 'sec elapsed')
print(grid_vox_spatial.shape)
# print(grid_spatial_vox)
# plt.imshow(grid_spatial_vox)
print("_______________________")


##############################################


print("Write input_data.npy in __pycache__")
res = np.sum(grid_vox, axis=2)
plt.imshow(res, cmap=cm.coolwarm)
np.save('__pycache__/input_data.npy', grid_vox)
# python scripts/visualize_dsi_volume.py -i input_data.npy
print("_______________________")

##############################################

print("Write input_data_spatial.npy in __pycache__")
res_spatial = np.sum(grid_vox_spatial, axis=2)
plt.imshow(res_spatial, cmap=cm.coolwarm)
np.save('__pycache__/input_data_spatial.npy', grid_vox_spatial)
# python scripts/visualize_dsi_volume.py -i input_data.npy
print("_______________________")

##############################################

print("Copying kernel data to the folder ", params.kernel_path)
for i in range(len(filters)):
    filename = params.kernel_path+'/kernel' + \
        str(i) + '.npy'  # adjust folder if needed
    np.save(filename, filters[i])
    # python scripts/visualize_dsi_volume.py -i file.npy
print("_______________________")

##############################################

print("Copying kernel data to the folder (spatial) ", params.kernel_path)
for i in range(len(filters_spatial)):
    filename = params.kernel_path+'/kernel' + \
        str(i) + 'spatial.npy'  # adjust folder if needed
    np.save(filename, filters_spatial[i])
    # python scripts/visualize_dsi_volume.py -i file.npy
print("_______________________")

##############################################

# 3D convolution with filter bank
print(" 3D convolution with filter bank ")
print("Number of filters", len(filters))
outs = []
with Timer("Convolustion..."):
    for filt in filters:
        out = signal.convolve(grid_vox, filt, mode='same')
        outs.append(out)
print("_______________________")

##############################################

# 3D convolution with filter bank spatial
print(" 3D convolution with filter bank (spatial)")
print("Number of filters (spatial)", len(filters_spatial))
outs_spatial = []
with Timer("Convolustion..."):
    for filt_spatial in filters_spatial:
        out_spatial = signal.convolve(grid_vox_spatial, filt_spatial, mode='same')
        outs_spatial.append(out_spatial)
print("_______________________")

##############################################

# Visualize results
print("Visualize results")
fig = plt.figure()
plt.figure(figsize=(24, 24))

max = 0
out_xy = []
for i in range(len(outs)):
    out_ = np.sum(np.abs(outs[i]), 2)
    max = np.max([max, np.max(out_)])
    out_xy.append(out_)

for i in range(len(outs)):
    plt.subplot(1, len(outs), i+1)
    plt.imshow(out_xy[i], vmax=0.8*max, vmin=0)
    plt.title('Output of filter ' + str(i))

# plt.grid()
# plt.show()
plt.savefig("../output_figures/Output_of_the_filters.png")
print("Output of ", str(i+1), " filters ")
print("_______________________")

##############################################

# Visualize results
print("Visualize results spatial")
fig = plt.figure()
plt.figure(figsize=(24, 24))

max_spatial = 0
out_xy_spatial = []
for i in range(len(outs_spatial)):
    out__spatial = np.sum(np.abs(outs_spatial[i]), 2)
    max_spatial = np.max([max_spatial, np.max(out__spatial)])
    out_xy_spatial.append(out__spatial)

for i in range(len(outs_spatial)):
    plt.subplot(1, len(outs_spatial), i+1)
    plt.imshow(out_xy_spatial[i], vmax=0.8*max_spatial, vmin=0)
    plt.title('Output of filter (Spatial) ' + str(i))

# plt.grid()
# plt.show()
plt.savefig("../output_figures/Output_of_the_filters_spatial.png")
print("Output of ", str(i+1), " filters (spatial) ")
print("_______________________")

##############################################

u = np.zeros((params.band_height, params.band_width), dtype=np.float32)
v = np.zeros((params.band_height, params.band_width), dtype=np.float32)
N = 8
with Timer("Aggregate..."):
    for k in range(len(filters)):
        u = u+np.cos(np.pi*2*k/N)*out_xy[k]
        v = v+(-1)*np.sin(np.pi*2*k/N)*out_xy[k]
print("u = ", u.shape)
print("v = ", v.shape)
print("_______________________")

fig = plt.figure(figsize=(24, 18))
image = np.zeros((params.band_height, params.band_width))
print("Image shape:")
print(image.shape)
plt.imshow(image, cmap='binary')
X, Y = np.meshgrid(np.arange(0, params.band_width),
                   np.arange(0, params.band_height))
plt.quiver(X, Y, u, v, color='r')
# for x_ in range(band_width):
#     for y_ in range(band_height):
#         plt.quiver(y_,x_,u[y_,x_],v[y_,x_],color='black',width=0.001,minlength=0.05)
plt.title('Output of velocity ')
plt.savefig("../output_figures/temporal_Output_of_velocity.png")
# plt.grid()
# plt.show()
print("_______________________")

##############################################
#Normalize

print('Normalize')
u = np.zeros((params.band_height, params.band_width), dtype=np.float32)
v = np.zeros((params.band_height, params.band_width), dtype=np.float32)
N = 8
with Timer("Aggregate..."):
    for k in range(len(filters)):
        u = u+np.cos(np.pi*2*k/N)*out_xy[k]
        v = v+(-1)*np.sin(np.pi*2*k/N)*out_xy[k]

u_normalized, v_normalized = normalize(u, v)
quiver_show_subset(u_normalized, v_normalized, 0, params.band_width, 0, params.band_height)
plt.savefig("../output_figures/whole_image_normalized.png")
print("_______________________")

##############################################


u_spatial = np.zeros((params.band_height, params.band_width), dtype=np.float32)
v_spatial = np.zeros((params.band_height, params.band_width), dtype=np.float32)
N_spatial = 8
with Timer("Aggregate..."):
    for k_spatial in range(len(filters_spatial)):
        u_spatial = u_spatial+np.cos(np.pi*2*k_spatial/N_spatial)*out_xy_spatial[k_spatial]
        v_spatial = v_spatial+(-1)*np.sin(np.pi*2*k_spatial/N_spatial)*out_xy_spatial[k_spatial]
print("u = ", u_spatial.shape)
print("v = ", v_spatial.shape)
print("_______________________")

fig = plt.figure(figsize=(24, 18))
image_spatial = np.zeros((params.band_height, params.band_width))
print("Image shape:")
print(image_spatial.shape)
plt.imshow(image_spatial, cmap='binary')
X_spatial, Y_spatial = np.meshgrid(np.arange(0, params.band_width),
                   np.arange(0, params.band_height))
plt.quiver(X_spatial, Y_spatial, u_spatial, v_spatial, color='r')
# for x_ in range(band_width):
#     for y_ in range(band_height):
#         plt.quiver(y_,x_,u[y_,x_],v[y_,x_],color='black',width=0.001,minlength=0.05)
plt.title('Output of velocity (spatial)')
plt.savefig("../output_figures/temporal_Output_of_velocity_spatial.png")
# plt.grid()
# plt.show()
print("_______________________")

##############################################

Nx_dft = 2*kernel_size
kx = np.arange(-Nx_dft/2, Nx_dft/2)*2./Nx_dft
x = np.arange(0, Nx_dft)  # pixels
f0 = 0.1  # cycles / pix
z = np.sin(2*np.pi*f0*x)
plt.plot(x, z)

##############################################

# Fourier design
print("Fourier design calculating:")

x_ = range(0, Nx_dft)
xv, yv = np.meshgrid(x_, x_)
z = np.sin(2*np.pi*(params.f0x_new*xv+params.f0y*yv))
Z = np.fft.fft2(z)
print(x_)
plt.imshow(np.log(1.+np.abs(Z)), cmap='gray')
plt.colorbar()
print("_______________________")

##############################################

Nt_dft = 256
F_tmono = np.fft.fft(Tmono, Nt_dft)
F_tbi = np.fft.fft(Tbi, Nt_dft)
kt = ft_nyquist * np.arange(-Nt_dft/2, Nt_dft/2)/Nt_dft
plt.plot(kt, np.abs(np.fft.fftshift(F_tmono)))

##############################################

plt.figure(figsize=(12, 12))
plt.plot(kt, np.abs(np.fft.fftshift(F_tbi)))
freq = np.fft.fftfreq(Nt_dft, d=dt_res)
plt.grid()
kt = np.fft.fftshift(freq)

##############################################

idx = np.argmax(np.abs(np.fft.fftshift(F_tbi)))
print(idx)
print(kt[idx])
print("_______________________")

##############################################
f0x_should_be = 9.11 / 90
print("what f0x should be")
print(f0x_should_be)
