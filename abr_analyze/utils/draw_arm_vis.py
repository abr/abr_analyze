import numpy as np
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

class DrawArmVis():
    """
    A class for simplifying 3d plotting of arm and trajectories
    """

    def __init__(self, traj_color='r', link_color='y', joint_color='g',
            arm_color='k', linestyle='-'):
        '''
        Set plotting parameters for the instantiated class

        PARAMETERS
        ----------
        traj_color: string
            a matplotlib compatible color used in the plot_trajectory()
            function
        link_color: string
            a matplotlib compatible color used in the plot_arm() function for
            coloring the link centers of mass
        joint_color: string
            a matplotlib compatible color used in the plot_arm() function for
            coloring the joints
        arm_color: string
            a matplotlib compatible color used in the plot_arm() function for
            coloring the arm segments connecting the origin to the joints and
            end-effector
        linestyle: string
            a matplotlib compatible linestyle, used in the plot_trajectory()
            functions
        '''
        self.traj_color = traj_color
        self.link_color = link_color
        self.joint_color = joint_color
        self.arm_color = arm_color
        self.linestyle = linestyle

    def plot_arm(self, ax, joints_xyz, links_xyz, ee_xyz):
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
            ax.scatter(xyz[0], xyz[1], xyz[2], c=self.joint_color)#, marker=marker, s=s,

        for xyz in links_xyz:
            # plot joint location
            ax.scatter(xyz[0], xyz[1], xyz[2], c=self.link_color)#, marker=marker, s=s,

        origin = [0,0,0]
        joints_xyz = np.vstack((origin, joints_xyz))

        ax.plot(joints_xyz.T[0], joints_xyz.T[1], joints_xyz.T[2],
                c=self.arm_color)

        return ax

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
            c = self.traj_color
        if linestyle is None:
            linestyle = self.linestyle

        ax.plot(data[:, 0], data[:,1], data[:,2],
                color=c, linestyle=linestyle)
        return ax
