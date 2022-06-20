"""
Functions for processing data including: interpolating for even sampling,
calculating average and confidence intervals, scaling data, filtering data,
and comparing to an ideal trajectory
"""

import numpy as np
import scipy.interpolate

from abr_analyze.data_handler import DataHandler


def get_mean_and_ci(raw_data, n=3000, p=0.95):
    """
    Gets the mean and 95% confidence intervals of data *see Note
    NOTE: data has to be grouped along rows, for example: having 5 sets of
    100 data points would be a list of shape (5,100)
    """
    sample = []
    upper_bound = []
    lower_bound = []
    sets = np.array(raw_data).shape[0]  # pylint: disable=E1136
    data_pts = np.array(raw_data).shape[1]  # pylint: disable=E1136
    print("Mean and CI calculation found %i sets of %i data points" % (sets, data_pts))
    raw_data = np.array(raw_data)
    for i in range(data_pts):
        data = raw_data[:, i]
        index = int(n * (1 - p) / 2)
        samples = np.random.choice(data, size=(n, len(data)))
        r = [np.mean(s) for s in samples]
        r.sort()
        ci = r[index], r[-index]
        sample.append(np.mean(data))
        lower_bound.append(ci[0])
        upper_bound.append(ci[1])

    data = {"mean": sample, "lower_bound": lower_bound, "upper_bound": upper_bound}
    return data


def list_to_function(data, time_intervals):
    """
    Accepts a list of dart points and returns an interpolated function that
    can be used to get points at different time intervals

    Parameters
    ----------
    data: list of floats time x dimension
        the data to turn into a function
    time_intervals: list of floats
        the timesteps corresponding to the data (not cumulative time)
    """
    data = np.asarray(data)
    sample_times = np.cumsum(time_intervals)

    functions = []
    if data.ndim == 1:
        data = data.reshape(len(data), 1)

    for kk in range(data.shape[1]):
        interp = scipy.interpolate.interp1d(sample_times, data[:, kk])
        functions.append(interp)

    return functions


def interpolate_data(data, time_intervals, interpolated_samples):
    """
    Accepts data and interpolates to the specified number of points

    Parameters
    ----------
    data: list of floats time x dimension
        the data to turn into a function
    time_intervals: list of floats
        the timesteps corresponding to the data (not cumulative time)
    interpolated_samples: int
        the number of evenly distributed points along get interpolated data
        for
    """
    data = np.asarray(data)
    time_intervals = np.asarray(time_intervals)

    run_time = sum(time_intervals)
    sample_times = np.cumsum(time_intervals)
    # interpolate to even samples out
    data_interp = []
    # if our array is one dimensional, make sure to add a second
    # dimension to avoid errors in our loop
    if data.ndim == 1:
        data = data.reshape(len(data), 1)
    for kk in range(data.shape[1]):
        interp = scipy.interpolate.interp1d(sample_times, data[:, kk])
        data_interp.append(
            np.array(
                [
                    interp(t)
                    for t in np.linspace(
                        sample_times[0], run_time, interpolated_samples
                    )
                ]
            )
        )
    data_interp = np.array(data_interp).T

    return data_interp


def scale_data(data, baseline_low, baseline_high, scaling_factor=1):
    """
    Scale data to some baseline to get values from 0-1 relative
    to baseline times the scaling factor

    Parameters
    data: list of floats
        the data to be scaled
    baseline_low: list of floats
        the lower error baseline that will be the zero
        reference
    baseline_high: list of floats
        the higher error baseline that will be the one
        reference
    scaling_factor: float, Optional (Default: 1)
        a value to scale the final data by
    """
    # TODO: add check for whether or not to np.array -ize
    data = np.asarray(data)
    baseline_low = np.asarray(baseline_low)
    baseline_high = np.asarray(baseline_high)
    scaled_data = (data - baseline_low) / (baseline_high - baseline_low)
    scaled_data *= scaling_factor

    return scaled_data


def load_and_process(db_name, save_location, parameters, interpolated_samples=None):
    # TODO: move interpolated samples is None check out of interpolation
    # function and add it here, no sense in having it check in the function
    # and have it do nothing, should only call interpolate if interpolating
    """
    Loads the parameters from the save_location,
    returns a dictionary of the interpolated and sampled data

    NOTE: if interpolated_samples is set to None, the raw data will be
    returned without interpolation and sampling

    Parameters
    ----------
    db_name: string
        the database where the data is saved
    save_location: string
        points to the location in the hdf5 database to read from
    parameters: list of strings
        the parameters to load and interpolate
    interpolated_samples: positive int, Optional (Default=100)
        the number of samples to take (evenly) from the interpolated data
        if set to None, no interpolated or sampling will be done, the raw
        data will be returned
    """
    # load data from hdf5 database
    dat = DataHandler(db_name=db_name)
    data = dat.load(parameters=parameters, save_location=save_location)

    # print(f"{db_name=}")
    # print(f"{parameters=}")
    # print(f"{save_location=}")
    # If time is not passed in, create a range from 0 to the length of any
    # other parameter in the list that is not time. This assumes that any
    # data passed in at once will be of the same length

    data_len = len(data[list(data.keys())[0]])
    if "time" not in parameters:
        data["time"] = np.ones(data_len)

    total_time = np.sum(data["time"])
    dat = []

    # interpolate for even sampling and save to our dictionary
    if interpolated_samples is not None:
        for key in data:
            if key != "time":
                data[key] = interpolate_data(
                    data=data[key],
                    time_intervals=data["time"],
                    interpolated_samples=interpolated_samples,
                )
    else:
        interpolated_samples = data_len

    # since we are interpolating over time, we are not interpolating
    # the time data, instead evenly sample interpolated_samples from
    # 0 to the sum(time)
    data["cumulative_time"] = np.linspace(0, total_time, interpolated_samples)

    data["read_location"] = save_location
    return data


def calc_cartesian_points(robot_config, q):
    """
    Takes in a robot_config and a list of joint angles and returns the
    cartesian coordinates of the robots joints and link COM's

    Parameters
    ----------
    robot_config: instantiated abr_control robot config
        This is required to transform joint angles to cartesian coordinates
    q: list of joint angles (n_timesteps, n_joints)
        The list of recorded joint angles used to transform link centers of
        mass and joint positions to cartesian coordinates
    """
    assert robot_config is not None, "robot_config must be provided"
    if hasattr(robot_config, "xml_file"):
        mujoco_model = True
    else:
        mujoco_model = False

    joints_xyz = []
    links_xyz = []
    ee_xyz = []

    # loop through our arm joints over time
    for q_t in q:
        joints_t_xyz = []
        links_t_xyz = []

        if not mujoco_model:
            # loop through the kinematic chain of joints
            for ii in range(0, robot_config.N_JOINTS):
                joints_t_xyz.append(robot_config.Tx("joint%i" % ii, q=q_t))
            joints_t_xyz.append(robot_config.Tx("EE", q=q_t))

            # loop through the kinematic chain of links
            for ii in range(0, robot_config.N_LINKS):
                links_t_xyz.append(robot_config.Tx("link%i" % ii, q=q_t))
        else:
            # loop through the kinematic chain of joints
            for ii in range(0, robot_config.N_JOINTS):
                joints_t_xyz.append(
                    robot_config.Tx("joint%i" % ii, q=q_t, object_type="site")
                )
            joints_t_xyz.append(robot_config.Tx("EE", q=q_t))

            # loop through the kinematic chain of links
            for ii in range(0, robot_config.N_JOINTS + 1):
                if ii == 0:
                    name = "base_link"
                else:
                    name = f"link{ii}"
                links_t_xyz.append(robot_config.Tx(name, q=q_t))

        # append the cartesian coordinates of this time step to our list
        joints_xyz.append(joints_t_xyz)
        links_xyz.append(links_t_xyz)
        ee_xyz.append(robot_config.Tx("EE", q=q_t))

    return [np.array(joints_xyz), np.array(links_xyz), np.squeeze(ee_xyz)]
