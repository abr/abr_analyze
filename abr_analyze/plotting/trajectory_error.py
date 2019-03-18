"""
Loads the data from the specified save location (see Note) for the provided
test and returns a dict of data, used to plot the error in trajectory,
interpolated and sampled to the specified number of interpolated_samples,
differentiated to the the order specified by time_derivative, and low pass
filtered with alpha=filter_const.

NOTE: it is recommended to have some filtering if differentiating data to
help smooth out peaks

NOTE: the location passed in must have data with the following keys at
minimum

'ee_xyz': list of n x 3 cartesian coordinates for the end-effector
'time': list of n timesteps
'ideal_trajectory': the path followed during the reach to the target
"""

import numpy as np
from abr_analyze.data_handler import DataHandler
from abr_analyze.data_processor import DataProcessor
from abr_analyze.data_visualizer import DataVisualizer

class TrajectoryError():
    def __init__(self, db_name, time_derivative=0, filter_const=None,
                 interpolated_samples=100):
        '''
        PARAMETERS
        ----------
        db_name: string
            the name of the database to load data from
        interpolated_samples: positive int, Optional (Default=100)
            the number of samples to take (evenly) from the interpolated data
            if set to None, no interpolated or sampling will be done, the raw
            data will be returned
        time_derivative: int, Optional (Default: 0)
            0: position
            1: velocity
            2: acceleration
            3: jerk
        filter_const: float, Optional, (Default: None)
            None for no filtering, it is recommended to filter higher order
            errors
        '''
        # instantiate our data processor
        self.proc = DataProcessor()
        self.vis = DataVisualizer()
        self.dat = DataHandler(db_name)
        self.db_name = db_name
        self.time_derivative = time_derivative
        self.filter_const = filter_const
        self.interpolated_samples = interpolated_samples

    def statistical_error(self, save_location, ideal=None, sessions=1, runs=1,
                          save_data=True, regen=False):
        '''
        calls the calculate error function to get the trajectory for all runs
        and sessions specified at the save location and calculates the mean
        error and confidence intervals

        PARAMETERS
        ----------
        save_location: string
            location of data in database
        ideal: string, Optional (Default: None)
            This tells the function what trajectory to calculate the error
            relative to
            None: use the saved filter data at save_location
            if string: key of Nx3 data in database to use
        sessions: int, Optional (Default: 1)
            the number of sessions to calculate error for
        runs: int, Optional (Default: 1)
            the number of runs in each session
        save_data: boolean, Optional (Default: True)
            True to save data, this saves the error for each session
        regen: boolean, Optional (Default: False)
            True to regenerate data
            False to load data if it exists
        '''
        if regen is False:
            exists = self.dat.check_group_exists(
                '%s/statistical_error_%s'%(save_location, self.time_derivative))
            if exists:
                ci_errors = self.dat.load(
                    parameters=['mean', 'upper_bound', 'lower_bound', 'ee_xyz',
                                'ideal_trajectory', 'time', 'time_derivative',
                                'filter_const', 'read_location', 'error'],
                    save_location='%s/statistical_error_%i' % (
                        save_location, self.time_derivative))
                if len(ci_errors['mean']) > 0:
                    exists = True
                else:
                    exists = False
        else:
            exists = False

        if not exists:
            errors = []
            for session in range(sessions):
                session_error = []
                for run in range(runs):
                    print('%.3f processing complete...' %
                        (100*((run+1)+(session*runs)) / (sessions*runs)),
                        end='\r')
                    loc = '%s/session%03d/run%03d' % (save_location, session, run)
                    data = self.calculate_error(save_location=loc, ideal=ideal)
                    session_error.append(np.sum(data['error']))
                errors.append(session_error)

            ci_errors = self.proc.get_mean_and_ci(raw_data=errors)
            ci_errors['time_derivative'] = self.time_derivative
            ci_errors['filter_const'] = self.filter_const

            if save_data:
                self.dat.save(
                    data=ci_errors,
                    save_location='%s/statistical_error_%i' % (
                        save_location, self.time_derivative),
                    overwrite=True)

        return ci_errors

    def calculate_error(self, save_location, ideal=None):
        '''
        loads the ee_xyz data from save_location and compares it to ideal. If
        ideal is not passed in, it is assuemed that a filtered path planner is
        saved in save_location under the key 'ideal_trajectory' and will be used
        as the reference for the error calculation. The data is loaded, interpolated,
        differentiated, and filtered based on the parameters. The two norm
        error over time is returned.

        the following dict is returned
        data = {
            'ee_xyz': list of end-effector positions (n_timesteps, xyz),
            'ideal_trajectory': list of path planner positions (n_timesteps, xyz),
            'time': list of timesteps (n_timesteps),
            'time_derivative': int, the order of differentiation applied,
            'filter_const': float or None, the filter value used,
            'read_location': string, the location the raw data was loaded from,
            'error': the two-norm error between the end-effector trajectory and
                the path planner followed that run

        PARAMETERS
        ----------
        save_location: string
            location of data in database
        ideal: string, Optional (Default: None)
            This tells the function what trajectory to calculate the error
            relative to
            None: use the saved filter data at save_location
            if string: key of Nx3 data in database to use
        '''
        if ideal is None:
            ideal = 'ideal_trajectory'
        parameters = ['ee_xyz', 'time', ideal]

        # load and interpolate data
        data = self.proc.load_and_process(
            db_name=self.db_name,
            save_location=save_location,
            parameters=parameters,
            interpolated_samples=self.interpolated_samples)

        if ideal == 'ideal_trajectory':
            data['ideal_trajectory'] = data['ideal_trajectory'][:, :3]
        dt = np.sum(data['time']) / self.interpolated_samples

        # integrate data
        if self.time_derivative > 0:
            # set our keys that are able to be differentiated to avoid errors
            differentiable_keys = ['ee_xyz', 'ideal_trajectory']
            if ideal is not None:
                differentiable_keys.append(ideal)

            for key in data:
                if key in differentiable_keys:
                    # differentiate the number of times specified by
                    # time_derivative
                    for _ in range(0, self.time_derivative):
                        data[key][:, 0] = np.gradient(data[key][:, 0], dt)
                        data[key][:, 1] = np.gradient(data[key][:, 1], dt)
                        data[key][:, 2] = np.gradient(data[key][:, 2], dt)

        # filter data
        if self.filter_const is not None:
            for key in data:
                if key != 'time':
                    data[key] = self.proc.filter_data(
                        data=data[key],
                        alpha=self.filter_const)

        data['time_derivative'] = self.time_derivative
        data['filter_const'] = self.filter_const
        data['read_location'] = save_location

        error = self.proc.two_norm_error(
            trajectory=data['ee_xyz'],
            ideal_trajectory=data[ideal],
            dt=dt)
        data['error'] = error

        return data

    def plot(self, ax, save_location, step=-1, c=None, linestyle='--',
             label=None, loc=1, title='Trajectory Error to Path Planner'):

        data = self.dat.load(
            parameters=['mean', 'upper_bound', 'lower_bound'],
            save_location='%s/statistical_error_%i'%(
                save_location, self.time_derivative))
        self.vis.plot_mean_and_ci(
            ax=ax, data=data, c=c, linestyle=linestyle,
            label=label, loc=loc, title=title)
