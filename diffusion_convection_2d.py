#
# Created by Brendan Berg on 01.11.2019
#

import numpy as np
import matplotlib.pyplot as plt


conductivity = 100.0                # [W/(m*K)]
specific_heat_capacity = 1000.0     # [J/(kg*K)]
depth = 0.1                         # [m]
size_of_block_height = 2.5          # [m]
size_of_block_width = 5.0           # [m]
number_of_cells_height = 30         # [#]
number_of_cells_width = 60          # [#]
temp_west = 373.0                   # [K]
temp_east = 373.0                   # [K]
temp_south = 273.0                  # [K]
temp_north = 273.0                  # [K]
heat_source_per_volume = 1000.0     # [W/m^3]
velocity_x = -0.5                   # [m/s]
velocity_y = 1.0                    # [m/s]
density = 1.0                       # [kg/m^3]


# utility functions

def cell_to_index(_i, _j):
    global number_of_cells_width
    return _j * number_of_cells_width + _i


# lengths / areas / volumes in the mesh ([m] / [m^2] / [m^3])

x_faces_tmp = np.linspace(0, size_of_block_width, number_of_cells_width + 1)
x_faces = np.tile(x_faces_tmp, number_of_cells_height).reshape(number_of_cells_height, number_of_cells_width + 1)
y_faces_tmp = np.linspace(0, size_of_block_height, number_of_cells_height + 1)
y_faces = np.repeat(y_faces_tmp, number_of_cells_width).reshape(number_of_cells_height + 1, number_of_cells_width)

x_centroids = 0.5 * (x_faces[:, 0:-1] + x_faces[:, 1:])
y_centroids = 0.5 * (y_faces[0:-1, :] + y_faces[1:, :])

width_of_cell = x_faces[:, 1:] - x_faces[:, 0:-1]
height_of_cell = y_faces[1:, :] - y_faces[0:-1, :]

x_area = depth * height_of_cell
x_area = np.hstack([x_area, x_area[:, -1].reshape(number_of_cells_height, 1)])
y_area = depth * width_of_cell
y_area = np.vstack([y_area, y_area[-1, :]])

volume_of_cell = depth * np.multiply(width_of_cell, height_of_cell)

d_centroids_x_interior = x_centroids[:, 1:] - x_centroids[:, 0:-1]
d_centroids_x_boundary_left = (2 * (x_centroids[:, 0] - x_faces[:, 0])).reshape(number_of_cells_height, 1)
d_centroids_x_boundary_right = (2 * (x_faces[:, -1] - x_centroids[:, -1])).reshape(number_of_cells_height, 1)
d_centroids_x = np.hstack([d_centroids_x_boundary_left, d_centroids_x_interior, d_centroids_x_boundary_right])

d_centroids_y_interior = y_centroids[1:, :] - y_centroids[0:-1, :]
d_centroids_y_boundary_bottom = (2 * (y_centroids[0, :] - y_faces[0, :]))
d_centroids_y_boundary_top = (2 * (y_faces[-1, :] - y_centroids[-1, :]))
d_centroids_y = np.vstack([d_centroids_y_boundary_bottom, d_centroids_y_interior, d_centroids_y_boundary_top])

velocity_at_face_x = np.repeat(np.repeat(velocity_x, number_of_cells_height), number_of_cells_width + 1).reshape(
    number_of_cells_height, number_of_cells_width + 1)
velocity_at_face_y = np.repeat(np.repeat(velocity_y, number_of_cells_height + 1), number_of_cells_width).reshape(
    number_of_cells_height + 1, number_of_cells_width)


# coefficients

x_DA = np.multiply(x_area, np.divide(conductivity, d_centroids_x))
y_DA = np.multiply(y_area, np.divide(conductivity, d_centroids_y))

x_F = density * specific_heat_capacity * np.multiply(velocity_at_face_x, x_area)
y_F = density * specific_heat_capacity * np.multiply(velocity_at_face_y, y_area)
x_Fw = np.copy(x_F[:, 0:-1])
x_Fe = np.copy(x_F[:, 1:])
y_Fs = np.copy(y_F[0:-1, :])
y_Fn = np.copy(y_F[1:, :])
x_Fwmax = np.maximum(x_Fw, np.zeros(x_Fw.shape))
x_Femax = np.maximum(-1 * x_Fe, np.zeros(x_Fe.shape))
y_Fsmax = np.maximum(y_Fs, np.zeros(y_Fs.shape))
y_Fnmax = np.maximum(-1 * y_Fn, np.zeros(y_Fn.shape))

Sp = np.zeros([number_of_cells_height, number_of_cells_width])
Sp[:, 0] += -1 * (2 * np.copy(x_DA[:, 0]) + np.copy(x_Fwmax[:, 0]))  # west
Sp[:, -1] += -1 * (2 * np.copy(x_DA[:, -1]) + np.copy(x_Femax[:, -1]))  # east
Sp[0, :] += -1 * (2 * np.copy(y_DA[0, :]) + np.copy(y_Fsmax[0, :]))  # south
Sp[-1, :] += -1 * (2 * np.copy(y_DA[-1, :]) + np.copy(y_Fnmax[-1, :]))  # north

Su = heat_source_per_volume * volume_of_cell
Su[:, 0] += temp_west * (2 * np.copy(x_DA[:, 0]) + np.copy(x_Fwmax[:, 0]))  # west
Su[:, -1] += temp_east * (2 * np.copy(x_DA[:, -1]) + np.copy(x_Femax[:, -1]))  # east
Su[0, :] += temp_south * (2 * np.copy(y_DA[0, :]) + np.copy(y_Fsmax[0, :]))  # south
Su[-1, :] += temp_north * (2 * np.copy(y_DA[-1, :]) + np.copy(y_Fnmax[-1, :]))  # north

mW = np.copy(x_DA[:, 0:-1]) + np.copy(x_Fwmax)
mW[:, 0] = np.zeros(mW[:, 0].shape)

mE = np.copy(x_DA[:, 1:]) + np.copy(x_Femax)
mE[:, -1] = np.zeros(mE[:, -1].shape)

mS = np.copy(y_DA[0:-1, :]) + np.copy(y_Fsmax)
mS[0, :] = np.zeros(mS[0, :].shape)

mN = np.copy(y_DA[1:, :]) + np.copy(y_Fnmax)
mN[-1, :] = np.zeros(mN[-1, :].shape)

mP = np.copy(mW) + np.copy(mE) + np.copy(mS) + np.copy(mN) + \
     (np.copy(x_Fe) - np.copy(x_Fw)) + (np.copy(y_Fn) - np.copy(y_Fs)) - np.copy(Sp)


# generate matrix

M = np.zeros([number_of_cells_width * number_of_cells_height, number_of_cells_width * number_of_cells_height])
s = np.zeros(number_of_cells_width * number_of_cells_height)

for j in range(number_of_cells_height):
    for i in range(number_of_cells_width):
        index = cell_to_index(i, j)

        M[index, index] = mP[j, i]
        s[index] = Su[j, i]

        if i != 0:
            west_index = cell_to_index(i - 1, j)
            M[index][west_index] = -mW[j, i]

        if i != (number_of_cells_width - 1):
            east_index = cell_to_index(i + 1, j)
            M[index][east_index] = -mE[j, i]

        if j != 0:
            south_index = cell_to_index(i, j - 1)
            M[index][south_index] = -mS[j, i]

        if j != (number_of_cells_height - 1):
            north_index = cell_to_index(i, j + 1)
            M[index][north_index] = -mN[j, i]

t = np.linalg.solve(M, s).reshape([number_of_cells_height, number_of_cells_width])


# plot

fig, ax = plt.subplots()
im = ax.pcolormesh(x_faces_tmp, y_faces_tmp, t)
fig.colorbar(im)

ax.axis('tight')
plt.show()
