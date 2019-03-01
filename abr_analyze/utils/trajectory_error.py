from abr_analyze.utils import DataHandler, DataProcessor, DataVisualizer
import numpy as np

class TrajectoryError():
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
    def __init__(self, db_name, time_derivative=0, filter_const=None,
            interpolated_samples=100, clear_memory=True):
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
        clear_memory: boolean, Optional (Default: True)
            True: overwrite the instantiated objects to None to save memory if
            looping through runs or tests, instead of keeping several
            instantiations of the database, dataprocessor, and robot_config objects
            False: leave the database, dataprocessor, and robot_config objects
            in case they need to be referenced.

        '''
        #NOTE: clear_memory is currently unused
        # instantiate our data processor
        self.proc = DataProcessor()
        self.dat = DataHandler(db_name)
        self.db_name = db_name
        self.time_derivative = time_derivative
        self.filter_const = filter_const
        self.interpolated_samples = interpolated_samples
        self.clear_memory = clear_memory

    def statistical_error(self, save_location, ideal=None, sessions=1, runs=1,
            save_data=True):
        '''

        '''
        errors = []
        for session in range(sessions)
            session_error = []
            for run in range(runs):
                loc = '%s/session%03d/run%03d'%(save_location, session, run)
                data = self.calculate_error(
                        save_location=save_location,
                        ideal=ideal)
                session_error.append(np.sum(data['error']))
            errors.append(session_error)
        print('error shape: ', np.array(errors).shape)
        ci_errors = self.proc.get_mean_and_ci(raw_data=errors)
        ci_errors['time_derivative'] = self.time_derivative
        ci_errors['filter_const'] = self.filter_const

        if save_data:
            self.dat.save(parameters=ci_errors,
                    save_location='%s/statistical_error_%i'%(save_location,
                        self.time_derivative))
        else:
            return ci_errors

    def calculate_error(self, save_location, ideal=None):
        '''
        loads the ee_xyz data from save_location and compares it to ideal. If
        ideal is not passed in, it is assuemed that a filtered path planner is
        saved in save_location under the key 'filter' and will be used as the
        reference for the error calculation. The data is loaded, interpolated,
        differentiated, and filtered based on the parameters. The two norm
        error over time is returned.

        the following dict is returned
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
        ideal: string, Optional (Default: None)
            This tells the function what trajectory to calculate the error
            relative to
            None: use the saved filter data at save_location
            if string: key of Nx3 data in database to use
        '''
        if ideal is None:
            ideal = 'filter'
        parameters = ['ee_xyz', ideal]

        # load and interpolate data
        data = self.proc.load_and_process(
                db_name=self.db_name,
                save_location=save_location,
                parameters=parameters,
                interpolated_samples=self.interpolated_samples)

        if ideal == 'filter':
            data['filter'] = data['filter'][:, :3]
        dt = np.mean(np.diff(data['time']))

        # integrate data
        if self.time_derivative > 0:
            for key in data:
                if key != 'time':
                    # differentiate the number of times specified by
                    # time_derivative
                    for ii in range(0, self.time_derivative):
                        print(np.array(data[key][:,0]).shape)
                        print(np.mean(data['time']))
                        data[key][:,0] = np.gradient(data[key][:,0], dt)
                        data[key][:,1] = np.gradient(data[key][:,1], dt)
                        data[key][:,2] = np.gradient(data[key][:,2], dt)

        # filter data
        if self.filter_const is not None:
            for key in data:
                if key != 'time':
                    data[key] = self.proc.filter_data(data=data[key],
                            alpha=self.filter_const)

        data['time_derivative'] = self.time_derivative
        data['filter_const'] = self.filter_const
        data['read_location'] = save_location

        error = self.two_norm_error(trajectory=data['ee_xyz'],
                ideal_trajectory=data[ideal], dt=dt)
        data['error'] = error

        return data

    def plot(self, ax, save_location, step=-1, c=None, linestyle='--',
            label=None, title='Trajectory Error to Path Planner'):
        data = self.dat.load(parameters=['mean', 'upper_bound', 'lower_bound'],
                save_location='%s/statistical_error_%i'%(save_location,
                    self.tiem_derivative))
