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

    # def clear_ax(self):
    #     '''
    #     clears all of the ax objects that have been passed in, this is useful
    #     when animating figures into a gif. Clearing the ax between frames
    #     greatly reduces the load and speeds up the processing
    #     '''
    #     for ax in self.master_ax:
    #         ax.clear()
    #     self.master_ax = []

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
                # self.master_ax.append(plt.subplot(inner_grid[row,col]))
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
        # self.master_ax.append(ax)
        return ax

    def plot_arm(self, ax, joints_xyz, links_xyz, ee_xyz, link_color='y',
            joint_color='k', arm_color='k'):
        '''
        accepts joint, end-effector, and link COM cartesian locations for one
        point in time and plots them on the provided 3D ax object

        returns the ax object

        PARAMETERS
        ----------
        'ee_xyz': np.array([x,y,z])
            cartesian coordinates of the end-effector
        'joints_xyz': np.array() (n_joints, 3 cartesian coordinates)
            cartesian coordinates of the joints
        'links_xyz': np.array() (n_links, 3 cartesian coordinates)
            cartesian coordinates of the link COM's

        EX: j for joint, t for time
            np.array([j0x_t0, j0y_t0, j0z_t0]),...,np.array(jNx_t0,
            jNy_t0, jNz_t0)]
        '''
        for xyz in joints_xyz:
            # plot joint location
            ax.scatter(xyz[0], xyz[1], xyz[2], c=joint_color)#, marker=marker, s=s,

        for xyz in links_xyz:
            # plot joint location
            ax.scatter(xyz[0], xyz[1], xyz[2], c=link_color)#, marker=marker, s=s,

        origin = [0,0,0]
        joints_xyz = np.vstack((origin, joints_xyz))

        ax.plot(joints_xyz.T[0], joints_xyz.T[1], joints_xyz.T[2],
                c=arm_color)
        # self.master_ax.append(ax)
        return ax

    def plot_data(self, ax, y, x=None, c=None, linestyle=None):
        if c is None:
            c = 'r'
        if linestyle is None:
            linestyle = '-'
        if x is None:
            ax.plot(y)
        else:
            ax.plot(x,y)
        return ax
