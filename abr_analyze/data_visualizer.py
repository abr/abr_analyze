"""
Functions plotting data onto ax objects
"""

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d  # pylint: disable=W0611
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def plot_arm(
    ax,
    joints_xyz,
    links_xyz,
    ee_xyz,
    link_color="y",
    joint_color="k",
    arm_color="k",
    title=None,
):
    """
    Accepts joint, end-effector, and link COM cartesian locations, and an
    ax object, returns a stick arm with points at the joints and link COM's
    plotted on the ax

    Parameters
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
    title: string, Optional (Default: None)
        the ax title
    """

    if isinstance(ax, list):
        if len(ax) > 1:
            raise Exception(
                "multi axis plotting is currently not available" + " for 3d plots"
            )
        ax = ax[0]

    for xyz in joints_xyz:
        # plot joint location
        ax.scatter(xyz[0], xyz[1], xyz[2], c=joint_color)

    for xyz in links_xyz:
        # plot joint location
        ax.scatter(xyz[0], xyz[1], xyz[2], c=link_color)

    origin = [0, 0, 0]
    joints_xyz = np.vstack((origin, joints_xyz))

    ax.plot(joints_xyz.T[0], joints_xyz.T[1], joints_xyz.T[2], c=arm_color)

    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(0.5, 1.2)
    ax.set_aspect(1)
    if title is not None:
        ax.set_title(title)

    return ax


def plot_2d_data(ax, y, x=None, c="r", linestyle="-", label=None, loc=1, title=None):
    """
    Accepts a list of data to plot onto a 2d ax and returns the ax object

    NOTE: if y is multidimensional, and a list of ax objects is passed in,
    each dimension will be plotted onto it's respective ax object. If one
    ax object is passed in, all dimensions will be plotted on it

    Parameters
    ----------
    ax: ax object for plotting
        can be a single ax object or a list of them
    y: list of data to plot
    x: list of time points to plot along y
    c: matplotlib compatible color to use in plotting
    linestyle: matplotlib compatible linestyle to use
    label: string, Optional (Default: None)
        the legend label for the data
    title: string, Optional (Default: None)
        the title of the ax object
    loc: int, Optional (Default: 1)
        the legend location
    """
    # TODO: should c and linestyle be accepted as lists?
    # TODO: check if x and y are supposed to be lists or arrays
    # turn the ax object into a list if it is not already one
    ax = make_list(ax)
    # if we received one ax object, plot everything on it
    if len(ax) == 1:
        if x is None:
            ax[0].plot(y, label=label)
        else:
            ax[0].plot(x, y, label=label)
        ax = ax[0]
        if label is not None:
            ax.legend(loc=loc)
        if title is not None:
            ax.set_title(title)
    # if a list of ax objects is passed in, plot each dimension onto its
    # own ax
    # TODO: need to check that ax and y dims match
    else:
        if label is None:
            label = ""
        for ii, a in enumerate(ax):
            if x is None:
                if y.ndim > 2:
                    a.plot(y[:, ii], label="%s %i" % (label, ii))
                else:
                    a.plot(y, label="%s %i" % (label, ii))
            elif y.ndim > 1:
                a.plot(x, y[:, ii], label="%s %i" % (label, ii))
            else:
                a.plot(x, y[ii], label="%s %i" % (label, ii))
            a.legend(loc=loc)
        if title is not None:
            ax[0].set_title(title)

    return ax


def plot_3d_data(
    ax,
    data,
    c="tab:purple",
    linestyle="-",
    emphasize_end=True,
    label=None,
    loc=1,
    title=None,
):
    """
    accepts an ax object and an n x 3 array to plot a 3d trajectory and
    returns the data plotted on the ax

    Parameters
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
    label: string, Optional (Default: None)
        the legend label for the data
    title: string, Optional (Default: None)
        the title of the ax object
    loc: int, Optional (Default: 1)
        the legend location
    """
    if isinstance(ax, list):
        if len(ax) > 1:
            raise Exception("multi axis plotting not available for 3d plots")
        ax = ax[0]

    ax.plot(
        data[:, 0], data[:, 1], data[:, 2], color=c, linestyle=linestyle, label=label
    )
    if emphasize_end:
        ax.scatter(data[-1, 0], data[-1, 1], data[-1, 2], color=c)

    if label is not None:
        ax.legend(loc=loc)
    if title is not None:
        ax.set_title(title)
    return ax


def plot_mean_and_ci(ax, data, c=None, linestyle="--", label=None, loc=1, title=None):
    """
    accepts dict with keys upper_bound, lower_bound, and mean, and plots
    the mean onto the ax object with the upper and lower bounds shaded

    Parameters
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
    label: string, Optional (Default: None)
        the legend label for the data
    title: string, Optional (Default: None)
        the title of the ax object
    loc: int, Optional (Default: 1)
        the legend location
    """
    ax.fill_between(
        range(np.array(data["mean"]).shape[0]),  # pylint: disable=E1136
        data["upper_bound"],
        data["lower_bound"],
        color=c,
        alpha=0.5,
    )
    ax.plot(data["mean"], color=c, label=label, linestyle=linestyle)
    ax.set_title(title)
    # TODO fix the legend here
    # ax.legend(loc)
    return ax


def make_list(param):
    """
    converts param into a list if it is not already one
    returns param as a list

    Parameters
    ----------
    param: any parameter to be converted into a list
    """
    if not isinstance(param, list):
        param = [param]
    return param


def project_data(data, n_dims_project):
    """
    perform SVD and project the data into its top n_dims_project
    principle components

    Parameters
    ----------
    data: np.array
        a timesteps x dimensions array of data
    n_dims_project: int
        the number of principle components to plot into
    """
    U, S, _ = np.linalg.svd(data.T)
    PCs = np.dot(U, np.diag(S))

    return np.dot(data, PCs[:, :n_dims_project])


def plot_against_projection_2d(ax, data_project, data_plot):
    """
    accepts a T x d project array, and a T x 1 plot array. Performs SVD on
    the data_project array, projects the data into the top n_projected_dims
    principle component, and plots against data_plot

    Parameters
    ----------
    ax: axis object
        allows for control of the plot from outside of this function
    data_project: np.array
        the data to perform SVD on and project into its top PC
    data_plot: np.array
        the data to plot the projected data against (y-axis)
    """
    projected = project_data(data_project, 1)

    ax.plot(projected, data_plot)

    return ax


def make_axes_for_3d_plots(ax, colorbar_ax=False):
    """
    accepts in an axis, in the same space adds two more axes.
    The intention is to use this to plot a 3D plot as 3 plots:
    (x, y), (y, z), (z, y), and optionally a 4th axis for the colorbar.

    Parameters
    ----------
    ax: axis object
    colorbar_ax: boolean, Option (Default: False)
    """
    # make the subplots in this axis
    divider = make_axes_locatable(ax)
    axes = [ax]
    axes.append(divider.append_axes("right", size="100%", pad="40%"))
    axes.append(divider.append_axes("right", size="100%", pad="40%"))
    if colorbar_ax:
        axes.append(divider.append_axes("right", size="7%", pad="10%"))

    return axes


def plot_against_projection_3d(ax, data_project, data_plot):
    """
    accepts a T x d project array, and a T x 1 plot array. Performs SVD on
    the data_project array, projects the data into the top 2 principle
    components, and plots against data_plot on the z axis

    Parameters
    ----------
    ax: axis object
        allows for control of the plot from outside of this function
    data_project: np.array
        the data to perform SVD on and project into its top PCs
    data_plot: np.array
        the data to plot the projected data against (z-axis)
    """
    projected = project_data(data_project, 2)
    axes = make_axes_for_3d_plots(ax)

    axes[0].plot(projected[:, 0], projected[:, 1])
    axes[1].plot(projected[:, 0], data_plot)
    axes[2].plot(projected[:, 1], data_plot)

    axes[0].update({"title": "XY"})
    axes[1].update({"title": "XZ"})
    axes[2].update({"title": "YZ"})

    return axes


def plot_against_projection_4d(ax, data_project, data_plot):
    """
    accepts a T x d project array, and a T x 1 plot array. Performs SVD on
    the data_project array, projects the data into the top three
    principle components, and uses data_plot to determine the color of
    each point

    Parameters
    ----------
    ax: axis object
        allows for control of the plot from outside of this function
    data_project: np.array
        the data to perform SVD on and project into its top PCs
    data_plot: np.array
        specifies the colors of the projected data points
    """
    projected = project_data(data_project, 3)
    axes = make_axes_for_3d_plots(ax, colorbar_ax=True)

    jet = plt.get_cmap("jet")
    cNorm = colors.Normalize(vmin=min(data_plot), vmax=max(data_plot))

    axes[0].scatter(projected[:, 0], projected[:, 1], c=data_plot, cmap="jet")
    axes[1].scatter(projected[:, 0], projected[:, 2], c=data_plot, cmap="jet")
    axes[2].scatter(projected[:, 1], projected[:, 2], c=data_plot, cmap="jet")
    matplotlib.colorbar.ColorbarBase(axes[3], cmap=jet, norm=cNorm)

    axes[0].update({"title": "XY"})
    axes[1].update({"title": "XZ"})
    axes[2].update({"title": "YZ"})

    return axes
