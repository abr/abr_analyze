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

        PARAMETERS
        ----------
        cell: gridspec.GridSpec(n,m)[x,y] object
        n_rows: int
            the number of rows to break the cell into
        n_cols: int
            the number of columns to break the cell into
        '''
        inner_grid = gridspec.GridSpecFromSubplotSpec(rows, cols, cell)
        ax = []
        for row in range(0, rows):
            for col in range(0, cols):
                ax.append(plt.subplot(inner_grid[row,col]))
        return ax

    def plot_arm(self, ax, joints_xyz, links_xyz, ee_xyz, link_color='y',
            joint_color='k', arm_color='k', title=None):
        '''
        accepts joint, end-effector, and link COM cartesian locations, and an
        ax object, returns a stick arm with points at the joints and link COM's
        plotted on the ax

        PARAMETERS
        ----------
        ax: ax object for plotting
        ee_xyz: np.array([x,y,z])
            cartesian coordinates of the end-effector
        joints_xyz: np.array() (n_joints, 3 cartesian coordinates)
            cartesian coordinates of the joints
        links_xyz: np.array() (n_links, 3 cartesian coordinates)
            cartesian coordinates of the link COM's
        link_color: matplotlib compatible color, Optional (Default: 'y')
            the color for the link center of mass points
        joint_color: matplotlib compatible color, Optional (Default: 'k')
            the color for the joint points
        arm_color: matplotlib compatible color, Optional (Default: 'k')
            the color for the links joining the joints
        '''

        if isinstance(ax, list):
            if len(ax) > 1:
                raise Exception("multi axis plotting is currently not availabe"
                        +" for 3d plots")
            else:
                ax = ax[0]

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

        ax.set_xlim3d(-0.5,0.5)
        ax.set_ylim3d(-0.5,0.5)
        ax.set_zlim3d(0.5,1.2)
        ax.set_aspect(1)
        if title is not None:
            ax.set_title(title)

        return ax

    def plot_2d_data(self, ax, y, x=None, c='r', linestyle='-', label=None,
            loc=1, title=None):
        '''
        Accepts a list of data to plot onto a 2d ax and returns the ax object

        NOTE: if y is multidimensional, and a list of ax objects is passed in,
        each dimension will be plotted onto it's respective ax object. If one
        ax object is passed in, all dimensions will be plotted on it

        PARAMETERS
        ----------
        ax: ax object for plotting
        y: list of data to plot
        x: list of time points to plot along y
        c: matplotlib compatible color to use in plotting
        linestyle: matplotlib compatible linestyle to use
        '''
        #TODO: should c and linestyle be accepted as lists?
        #TODO: check if x and y are supposed to be lists or arrays
        # turn the ax object into a list if it is not already one
        ax = self.make_list(ax)
        # if we received one ax object, plot everything on it
        if len(ax) == 1:
            if x is None:
                ax[0].plot(y, label=label)
            else:
                ax[0].plot(x, y, label=label)
            ax = ax[0]
            ax.legend(loc=loc)
            if title is not None:
                ax.set_title(title)
        # if a list of ax objects is passed in, plot each dimension onto its
        # own ax
        #TODO: need to check that ax and y dims match
        else:
            if label is None:
                label = ''
            for ii, a in enumerate(ax):
                if x is None:
                    a.plot(y[:, ii], label='%s %i'%(label, ii))
                else:
                    a.plot(x,y[:,ii], label='%s %i'%(label,ii))
                a.legend(loc=loc)
            if title is not None:
                ax[0].set_title(title)

        return ax

    def plot_3d_data(self, ax, data, c='tab:purple', linestyle='-', emphasize_end=True,
            label=None, loc=1, title=None):
        '''
        accepts an ax object and an n x 3 aray to plot a 3d trajectory and
        returns the data plotted on the ax

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
        emphasize_end: boolean, Optional (Default: True)
            True to add a point at the final position with a larger size
        '''
        if isinstance(ax, list):
            if len(ax) > 1:
                raise Exception("multi axis plotting is currently not availabe"
                        +" for 3d plots")
            else:
                ax = ax[0]

        ax.plot(data[:, 0], data[:,1], data[:,2],
                color=c, linestyle=linestyle, label=label)
        if emphasize_end:
            ax.scatter(data[-1,0], data[-1,1], data[-1,2],
                color=c)

        if label is not None:
            ax.legend(loc=loc)
        if title is not None:
            ax.set_title(title)
        return ax

    def plot_mean_and_ci(self, ax, data, c=None, linestyle='-', label=None,
            loc=1, title=None):
        '''

        '''
        ax.fill_between(range(np.array(data['mean']).shape[0]),
                         data['upper_bound'],
                         data['lower_bound'],
                         color=c,
                         alpha=.5)
        ax.plot(data['mean'], color=c, label=label, linestyle='--')
        ax.set_title(title)
        #TODO fix the legend here
        #ax.legend(loc)
        return ax


    def make_list(self, param):
        '''
        converts param into a list if it is not already one

        PARAMETERS
        ----------
        param: any parameter to be converted into a list
        '''
        if not isinstance(param, list):
            param = [param]
        return param
