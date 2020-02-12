#
# Created by Brendan Berg on 01.11.2019
#

import numpy as np
import matplotlib.pyplot as plt


# input
from numpy.core.multiarray import ndarray

conductivity = 100.0                # [W/(m*K)]
specific_heat_capacity = 1000.0     # [J/(kg*K)]
depth = 0.1                         # [m]
size_of_block_height = 2.5          # [m]
size_of_block_width = 5.0           # [m]
number_of_cells_height = 30         # [#]
number_of_cells_width = 60          # [#]
temp_left = 100.0                   # [deg C]
temp_right = 100.0                  # [deg C]
temp_bottom = 0.0                   # [deg C]
temp_top = 0.0                      # [deg C]
heat_source_per_volume = 1000.0     # [W/m^3]
velocity_x = -0.5                   # [m/s]
velocity_y = 1.0                    # [m/s]
density = 1.0                       # [kg/m^3]


# utility functions

def cell_to_index(_i, _j):
    global number_of_cells_width
    return _j * number_of_cells_width + _i


# lengths / volumes in the mesh ([m] and [m^3])

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
x_Fl = np.copy(x_F[:, 0:-1])
x_Fr = np.copy(x_F[:, 1:])
y_Fb = np.copy(y_F[0:-1, :])
y_Ft = np.copy(y_F[1:, :])
x_Flmax = np.maximum(x_Fl, np.zeros(x_Fl.shape))
x_Frmax = np.maximum(-1 * x_Fr, np.zeros(x_Fr.shape))
y_Fbmax = np.maximum(y_Fb, np.zeros(y_Fb.shape))
y_Ftmax = np.maximum(-1 * y_Ft, np.zeros(y_Ft.shape))

Sp = np.zeros([number_of_cells_height, number_of_cells_width])
Sp[:, 0] += -1 * (2 * np.copy(x_DA[:, 0]) + np.copy(x_Flmax[:, 0]))  # left
Sp[:, -1] += -1 * (2 * np.copy(x_DA[:, -1]) + np.copy(x_Frmax[:, -1]))  # right
Sp[0, :] += -1 * (2 * np.copy(y_DA[0, :]) + np.copy(y_Fbmax[0, :]))  # bottom
Sp[-1, :] += -1 * (2 * np.copy(y_DA[-1, :]) + np.copy(y_Ftmax[-1, :]))  # top

Su = heat_source_per_volume * volume_of_cell
Su[:, 0] += temp_left * (2 * np.copy(x_DA[:, 0]) + np.copy(x_Flmax[:, 0]))  # left
Su[:, -1] += temp_right * (2 * np.copy(x_DA[:, -1]) + np.copy(x_Frmax[:, -1]))  # right
Su[0, :] += temp_bottom * (2 * np.copy(y_DA[0, :]) + np.copy(y_Fbmax[0, :]))  # bottom
Su[-1, :] += temp_top * (2 * np.copy(y_DA[-1, :]) + np.copy(y_Ftmax[-1, :]))  # top

aL = np.copy(x_DA[:, 0:-1]) + np.copy(x_Flmax)
aL[:, 0] = np.zeros(aL[:, 0].shape)

aR = np.copy(x_DA[:, 1:]) + np.copy(x_Frmax)
aR[:, -1] = np.zeros(aR[:, -1].shape)

aB = np.copy(y_DA[0:-1, :]) + np.copy(y_Fbmax)
aB[0, :] = np.zeros(aB[0, :].shape)

aT = np.copy(y_DA[1:, :]) + np.copy(y_Ftmax)
aT[-1, :] = np.zeros(aT[-1, :].shape)

aP = np.copy(aL) + np.copy(aR) + np.copy(aB) + np.copy(aT) +\
        (np.copy(x_Fr) - np.copy(x_Fl)) + (np.copy(y_Ft) - np.copy(y_Fb)) - np.copy(Sp)


# generate matrix

A = np.zeros([number_of_cells_width * number_of_cells_height, number_of_cells_width * number_of_cells_height])
B = np.zeros(number_of_cells_width * number_of_cells_height)

for j in range(number_of_cells_height):
    for i in range(number_of_cells_width):
        index = cell_to_index(i, j)

        A[index, index] = aP[j, i]
        B[index] = Su[j, i]

        if i != 0:
            left_index = cell_to_index(i - 1, j)
            A[index][left_index] = -aL[j, i]

        if i != (number_of_cells_width - 1):
            right_index = cell_to_index(i + 1, j)
            A[index][right_index] = -aR[j, i]

        if j != 0:
            bottom_index = cell_to_index(i, j - 1)
            A[index][bottom_index] = -aB[j, i]

        if j != (number_of_cells_height - 1):
            top_index = cell_to_index(i, j + 1)
            A[index][top_index] = -aT[j, i]

T = np.linalg.solve(A, B).reshape([number_of_cells_height, number_of_cells_width])


# plot

fig, ax = plt.subplots()
im = ax.pcolormesh(x_faces_tmp, y_faces_tmp, T)
fig.colorbar(im)

ax.axis('tight')
plt.show()
