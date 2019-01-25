from abr_analyze.utils import DataHandler, DataProcessor
import numpy as np

class TrajectoryErrorProc():
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
    'filter': the path followed during the reach to the target
    """
    def __init__(self, db_name):
        '''
        PARAMETERS
        ----------
        db_name: string
            the name of the database to load data from
        '''
        # instantiate our data processor
        self.proc = DataProcessor()
        # instantiate our database object
        self.dat = DataHandler(db_name=db_name)


    def load_and_process(self, save_location, time_derivative,
            filter_const=None, interpolated_samples=100):
        '''
        interpolates, integrates and filters the provided trajectory

        PARAMETERS
        ----------
        save_location: string
            location of data in database
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
        # load data
        data = self.dat.load(params=['ee_xyz', 'time', 'filter'],
                save_location=save_location)
        data['filter'] = data['filter'][:, :3]

        # interpolate data
        if interpolated_samples is not None:
            for key in data:
                if key != 'time':
                    data[key] = self.proc.interpolate_data(data=data[key],
                            time_intervals=data['time'],
                            n_points=interpolated_samples)
            # since we are interpolating over time, we are not interpolating
            # the time data, instead evenly sample interpolated_samples from
            # 0 to the sum(time)
            data['time'] = np.linspace(0, sum(data['time']),
                    interpolated_samples)
            dt = np.mean(np.diff(data['time']))

        # integrate data
        if time_derivative > 0:
            for key in data:
                if key != 'time':
                    # differentiate the number of times specified by
                    # time_derivative
                    for ii in range(0, time_derivative):
                        print(np.array(data[key][:,0]).shape)
                        print(np.mean(data['time']))
                        data[key][:,0] = np.gradient(data[key][:,0], dt)
                        data[key][:,1] = np.gradient(data[key][:,1], dt)
                        data[key][:,2] = np.gradient(data[key][:,2], dt)

        # filter data
        if filter_const is not None:
            for key in data:
                if key != 'time':
                    data[key] = self.proc.filter_data(data=data[key],
                            alpha=filter_const)

        data['time_derivative'] = time_derivative
        data['filter_const'] = filter_const
        data['read_location'] = save_location
        return data

    def two_norm_error(self, trajectory, ideal_trajectory, dt):
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
        error = (np.sum(np.sqrt(np.sum(
            (ideal_trajectory - trajectory)**2,
            axis=1)))) *dt
        #TODO: confirm whether or not we should be multiplying by dt

        return error

    def generate(self, save_location, time_derivative=0, filter_const=None,
            interpolated_samples=100, clear_memory=True):
        '''
        Loads the relevant test data to compare trajectories, saving all to
        a dictionary of interpolated data for even sampling between tests.
        Returns a dict of the data * see note

        data = {
            'ee_xyz': list of end-effector positions (n_timesteps, xyz),
            'filter': list of path planner positions (n_timesteps, xyz),
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
        clear_memory: boolean, Optional (Default: True)
            True: overwrite the instantiated objects to None to save memory if
            looping through runs or tests, instead of keeping several
            instantiations of the database, dataprocessor, and robot_config objects
            False: leave the database, dataprocessor, and robot_config objects
            in case they need to be referenced.
        '''
        # load and process our data, including interpolation, differentiation,
        # and filtering, dependant on the user inputs
        data = self.load_and_process(save_location=save_location,
                time_derivative=time_derivative, filter_const=filter_const,
                interpolated_samples=interpolated_samples)

        error = self.two_norm_error(trajectory=data['ee_xyz'],
                ideal_trajectory=data['filter'], dt=np.mean(data['time']))
        data['error'] = error

        return data
