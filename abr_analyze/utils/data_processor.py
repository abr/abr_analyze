"""
Class for processing data including: interpolating for even sampling,
calculating average and confidence intervals, scaling data, filtering data, and
comparing to an ideal trajectory

Process for comparing to ideal trajectory
1. interpolate data for even sampling
2. generate ideal trajectory with the same sampling
3. calculate error from recorded path to ideal
4. filter data if desired (recommended for 2nd and 3rd order error)
5. scale to a baseline if desired
"""
import numpy as np
import os
import scipy.interpolate

from abr_analyze.utils.paths import cache_dir
from abr_analyze.utils import DataHandler

class DataProcessor():
    def __init__(self):
        """
        """
        pass

    def get_mean_and_ci(self, raw_data):
        '''
        gets the mean and 95% confidence intervals of data *see Note

        NOTE: data has to be grouped along rows, for example: having 5 sets of
        100 data points would be a list of shape (5,100)
        '''
        sample = []
        upper_bound = []
        lower_bound = []
        sets = np.array(raw_data).shape[0]
        data_pts = np.array(raw_data).shape[1]
        print('Mean and CI calculation found %i sets of %i data points'
                %(sets, data_pts))
        raw_data = np.array(raw_data)
        for i in range(data_pts):
            data = raw_data[:,i]
            ci = self.bootstrapci(data, np.mean)
            sample.append(np.mean(data))
            lower_bound.append(ci[0])
            upper_bound.append(ci[1])

        data = {'mean': sample, 'lower_bound': lower_bound, 'upper_bound':
                upper_bound}
        return data

    def bootstrapci(self, data, func, n=3000, p=0.95):
        index=int(n*(1-p)/2)
        samples = np.random.choice(data, size=(n, len(data)))
        r = [func(s) for s in samples]
        r.sort()
        return r[index], r[-index]

    def list_to_function(self, data, time_intervals):
        time_intervals = np.cumsum(time_intervals)
        functions = []
        for kk in range(data.shape[1]):
            interp = scipy.interpolate.interp1d(time_intervals, data[:, kk])
            functions.append(interp)

        return functions

    def interpolate_data(self, data, time_intervals, interpolated_samples):
        if interpolated_samples is not None:
            run_time = sum(time_intervals)
            time_intervals = np.cumsum(time_intervals)
            # interpolate to even samples out
            data_interp = []
            # if our array is one dimensional, make sure to add a second dimension
            # to avoid errors in our loop
            if data.ndim == 1:
                data = data.reshape(len(data), 1)
            for kk in range(data.shape[1]):
                interp = scipy.interpolate.interp1d(time_intervals, data[:, kk])
                data_interp.append(np.array([
                    interp(t) for t in np.linspace(time_intervals[0], run_time,
                        interpolated_samples)]))
            data_interp = np.array(data_interp).T
        else:
            data_interp = data

        return data_interp

    def scale_data(self, input_data, baseline_low, baseline_high, scaling_factor=1):
        """
        scale data to some baseline to get values from 0-1 relative
        to baseline times the scaling factor

        PARAMETERS
        input_data: list of floats
            the data to be scaled
        baseline_low: list of floats
            the lower error baseline that will be the zero
            reference
        baseline_high: list of floats
            the higher error baseline that will be the one
            reference
        """
        #TODO: add check for whether or not to np.array -ize
        input_data = np.array(input_data)
        baseline_low = np.array(baseline_low)
        baseline_high = np.array(baseline_high)
        scaled_data = ((input_data - baseline_low)
                       / (baseline_high - baseline_low))
        scaled_data *= scaling_factor

        return scaled_data

    def filter_data(self, data, alpha=0.2):
        data_filtered = []
        for nn in range(0,len(data)):
            if nn == 0:
                data_filtered.append((alpha*data[nn]
                                          + (1-alpha)*0))
            else:
                data_filtered.append((alpha*data[nn]
                                          + (1-alpha)*data_filtered[nn-1]))
        data_filtered = np.array(data_filtered)

        return data_filtered

    def load_and_process(self, db_name, save_location, parameters,
            interpolated_samples=100):
        """
        Loads the relevant data for 3d arm plotting from the save_location,
        returns a dictionary of the interpolated and sampled data

        NOTE: if interpolated_samples is set to None, the raw data will be return without
        interpolation and sampling

        PARAMETERS
        ----------
        save_location: string
            points to the location in the hdf5 database to read from
        interpolated_samples: positive int, Optional (Default=100)
            the number of samples to take (evenly) from the interpolated data
            if set to None, no interpolated or sampling will be done, the raw
            data will be returned
        """
        # load data from hdf5 database
        dat = DataHandler(db_name=db_name)
        data = dat.load(parameters=parameters,
                save_location=save_location)

        # If time is not passed in, create a range from 0 to the length of any
        # other parameter in the list that is not time. This assumes that any
        # data passed in at once will be of the same length
        if 'time' not in parameters:
            for param in parameters:
                data['time']=range(0, len(data[param]))
                break
        # elif data['time'].item() is None:
        #     for param in parameters:
        #         if param is not 'time':
        #             data['time']=range(0, len(data[param]))
        #             break

        total_time = np.sum(data['time'])
        dat = []

        # interpolate for even sampling and save to our dictionary
        for key in data:
            if key != 'time':
                data[key] = self.interpolate_data(
                    data=data[key],
                    time_intervals=data['time'],
                    interpolated_samples=interpolated_samples)
        # since we are interpolating over time, we are not interpolating
        # the time data, instead evenly sample interpolated_samples from
        # 0 to the sum(time)
        if interpolated_samples is None:
            interpolated_samples = len(data[key])
        data['time'] = np.linspace(0, total_time, interpolated_samples)

        data['read_location'] = save_location
        return data

    def calc_cartesian_points(self, robot_config, q):
        """
        Takes in a robot_config and a list of joint angles and returns the
        cartesian coordinates of the robots joints and link COM's

        PARAMETERS
        ----------
        robot_config: instantiated abr_control robot config
            This is required to transform joint angles to cartesian coordinates
        q: list of joint angles (n_timesteps, n_joints)
            The list of recorded joint angles used to transform link centers of
            mass and joint positions to cartesian coordinates
        """
        assert robot_config is not None, 'robot_config must be provided'

        joints_xyz = []
        links_xyz = []
        ee_xyz = []

        # loop through our arm joints over time
        for q_t in q:
            joints_t_xyz= []
            links_t_xyz = []
            ee_t_xyz = []

            # loop through the kinematic chain of joints
            for ii in range(0, robot_config.N_JOINTS):
                joints_t_xyz.append(robot_config.Tx('joint%i'%ii, q=q_t))
            joints_t_xyz.append(robot_config.Tx('EE', q=q_t))
            ee_t_xyz.append(robot_config.Tx('EE', q=q_t))

            # loop through the kinematic chain of links
            for ii in range(0, robot_config.N_LINKS):
                links_t_xyz.append(robot_config.Tx('link%i'%ii, q=q_t))

            # append the cartesian coordinates of this time step to our list
            joints_xyz.append(joints_t_xyz)
            links_xyz.append(links_t_xyz)
            ee_xyz.append(ee_t_xyz)

        return [np.array(joints_xyz), np.array(links_xyz), np.squeeze(ee_xyz)]

    def two_norm_error(self, trajectory, ideal_trajectory, dt=1):
        """
        accepts two nx3 arrays of xyz cartesian coordinates and returns the
        2norm error of traj to baseline_traj

        Parameters
        ----------
        baseline_traj: nx3 array
            coordinates of ideal trajectory over time
        traj: nx3 array
            coordinates of trajectory to compare to baseline
        dt: float
            average timestep
        """
        # error relative to ideal path
        error = np.sqrt(np.sum(
            (ideal_trajectory - trajectory)**2,
            axis=1)) *dt
        #TODO: confirm whether or not we should be multiplying by dt

        return error


