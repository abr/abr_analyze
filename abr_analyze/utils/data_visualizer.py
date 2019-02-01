import numpy as np
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec
from mpl_toolkits.mplot3d import axes3d
# import subprocess
# import os

class DataVisualizer():
    def __init__(self):
        pass
    def cell_to_subplot(cell, n_rows, n_cols):
        '''
        Accepts a gridspec.Gridspec(n,m)[x,y] cell, and breaks that location
        up into n_rows and n_cols ax objects, returns a list of ax objects

        *EXAMPLE*
        outer_grid = gridspec.GridSpec(3,3)
        cell = outer_grid[1,2]
        ax = cell_to_subplot(cell=cell, n_rows=3, n_cols=1)
        ax[0].plot(data0)
        ax[1].plot(data1)
        ax[2].plot(data2)
        plt.show()
        '''
        inner_grid = gridspec.GridSpecFromSubplotSpec(rows, cols, cell)
        ax = []
        for row in range(0, rows):
            for col in range(0, cols):
                ax.append(plt.subplot(inner_grid[row,col]))
        return ax

    # def align_yaxis(ax1, v1, ax2, v2):
    #     """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    #     _, y1 = ax1.transData.transform((0, v1))
    #     _, y2 = ax2.transData.transform((0, v2))
    #     inv = ax2.transData.inverted()
    #     _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    #     miny, maxy = ax2.get_ylim()
    #     ax2.set_ylim(miny+dy, maxy+dy)

    def plot_trajectory(self, ax, data, c=None, linestyle=None):
        '''
        accepts an n x 3 aray to plot a 3d trajectory

        PARAMETERS
        ----------
        ax: axis object
            allows for control of the plot from outside of this function
        data: n x 3 array of 3D cartesian coordinates
        c: string, Optional (Default: None)
            matplotlib compatible color to be used when plotting data, this
            allows the user to overwrite the instantiated value in case the
            same instantiated DrawArmVis object is used for multiple trajectory
            plots
        linestyle: string, Optional (Default: None)
            matplotlib compatible linestyle to be used when plotting data, this
            allows the user to overwrite the instantiated value in case the
            same instantiated DrawArmVis object is used for multiple trajectory
            plots
        '''
        if c is None:
            c = 'tab:purple'
        if linestyle is None:
            linestyle = '-'

        ax.plot(data[:, 0], data[:,1], data[:,2],
                color=c, linestyle=linestyle)
        return ax


