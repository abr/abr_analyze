"""
Generate a random trajectory in 3d space and an ideal path.

Both are the same randomly generated path with different
amounts of filtering. This is mainly used to provide test data
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d  # pylint: disable=W0611


def generate(steps=100, plot=False):
    alpha = 0.7
    ee_xyz = [[np.random.uniform(0.05, 0.2, 1),
               np.random.uniform(0.05, 0.2, 1),
               np.random.uniform(0.5, 1.0, 1)]]

    for ii in range(steps-1):

        if ii == 0:
            xx = np.random.uniform(-2, 2, 1)/100
            yy = np.random.uniform(-2, 2, 1)/100
            zz = np.random.uniform(-2, 2, 1)/100
            x = ee_xyz[-1][0] + xx
            y = ee_xyz[-1][1] + yy
            z = ee_xyz[-1][2] + zz

        else:
            xx = xx * alpha + np.random.uniform(-2, 2, 1)/100 * (1-alpha)
            yy = yy * alpha + np.random.uniform(-2, 2, 1)/100 * (1-alpha)
            zz = zz * alpha + np.random.uniform(-2, 2, 1)/100 * (1-alpha)
            x = ee_xyz[-1][0] + xx
            y = ee_xyz[-1][1] + yy
            z = ee_xyz[-1][2] + zz

        ee_xyz.append([x, y, z])
    ee_xyz = np.squeeze(np.array(ee_xyz))

    alpha = 0.2
    ideal = np.zeros((steps, 3))
    for ii, val in enumerate(ee_xyz.tolist()):

        if ii == 0:
            ideal[0] = val

        else:
            ideal[ii][0] = alpha*val[0] + (1-alpha)*ideal[ii-1][0]
            ideal[ii][1] = alpha*val[1] + (1-alpha)*ideal[ii-1][1]
            ideal[ii][2] = alpha*val[2] + (1-alpha)*ideal[ii-1][2]

    ideal = np.array(ideal)
    times = np.ones(steps) * 0.03 + np.random.rand(steps)/50

    data = {'ee_xyz': ee_xyz, 'ideal_trajectory': ideal, 'time': times}

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('Generated Trajectory')
        ax.plot(ee_xyz[:, 0], ee_xyz[:, 1], ee_xyz[:, 2], label='ee_xyz')
        ax.plot(ideal[:, 0], ideal[:, 1], ideal[:, 2], label='ideal')
        ax.legend()
        plt.show()

    return data
