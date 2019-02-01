from abr_analyze.utils import DataHandler, DataProcessor
import numpy as np

class DrawArmProc():
    """
    Loads the data from the specified save location (see Note) for the provided
    test and returns a dict of data, used to plot a 3D visualization of the arm,
    interpolated and sampled to the specified number of interpolated_samples.

    NOTE: the location passed in must have data with the following keys at
    minimum
    'ee_xyz': list of n x 3 cartesian coordinates for the end-effector
    'target': list of n x 3 cartesian coordinates for the reaching target
    'time': list of n timesteps
    'filter': the path followed during the reach to the target
    * to draw an arm on a 3D plot you will also need...
    'q': list of n x N_JOINTS joint angles
    """
    def __init__(self, db_name, robot_config=None):
        """
        instantiate the required modules

        PARAMETERS
        ----------
        db_name: string
            name of the hdf5 database to load data from
        robot_config: instantiated abr_control robot config
            This is required to transform joint angles to cartesian coordinates
            if None only the trajectory of the end-effector will be processed
        """
        # instantiate our data processor
        self.proc = DataProcessor()
        # instantiate our database object
        self.dat = DataHandler(db_name=db_name)
        # generate any config functions that have not been generated and saved
        self.robot_config = robot_config

    def load_and_process(self, save_location, interpolated_samples=100):
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
        assert ((isinstance(interpolated_samples, int)
                 and interpolated_samples>0),
                ('TYPE ERROR: interpolated_samples must be a positive integer'
                    + ': received: sign(%i), type(%s)'%
                    (np.sign(interpolated_samples),
                        type(interpolated_samples))))

        # set params depending on whether a 3d arm will be plotted, or just the
        # end-effector trajectory
        params = ['ee_xyz', 'target', 'time', 'filter']

        if self.robot_config is not None:
            params.append('q')

        # load data from hdf5 database
        data = self.dat.load(params=params,
                save_location=save_location)

        # interpolate for even sampling and save to our dictionary
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

        # loop through our arm joints over time
        for q_t in q:
            joints_t_xyz= []
            links_t_xyz = []

            # loop through the kinematic chain of joints
            for ii in range(0, robot_config.N_JOINTS):
                joints_t_xyz.append(robot_config.Tx('joint%i'%ii, q=q_t,
                        x=robot_config.OFFSET))
            joints_t_xyz.append(robot_config.Tx('EE', q=q_t,
                x=robot_config.OFFSET))

            # loop through the kinematic chain of links
            for ii in range(0, robot_config.N_LINKS):
                links_t_xyz.append(robot_config.Tx('link%i'%ii, q=q_t,
                        x=robot_config.OFFSET))

            # append the cartesian coordinates of this time step to our list
            joints_xyz.append(joints_t_xyz)
            links_xyz.append(links_t_xyz)

        return [np.array(joints_xyz), np.array(links_xyz)]

    def check_limits(self):
        """
        Checks limits to note collisions with floor

        PARAMETERS
        ----------
        """
        raise Exception ("""The check_limits feature is currently not supported""")

    def generate(self, save_location, interpolated_samples=100, clear_memory=True):
        """
        Loads the relevant test data and calculates the required information
        for plotting a virtual arm, saving all to a dictionary of interpolated
        data for even sampling between tests. Returns a dict of the data *see
        note*

        data = {
            'ee_xyz': list of end-effector positions (n_timesteps, xyz),
            'target': list of target positions (n_timesteps, xyz),
            'filter': list of path planner positions (n_timesteps, xyz),
            'time': list of timesteps (n_timesteps)
            * the following are only returned if a robot_config is provided
            'q': list of joint angles (n_timesteps, n_joints)
            'joints_xyz': list of cartesian coordinates of the joints over time
            'links_xyz': list of cartesian coordinates of the link COM's over time
            }

            EX joints_xyz and links_xyz: j for joint, t for time
            np.array([
                [np.array([j0x_t0, j0y_t0, j0z_t0]),...,np.array(jNx_t0,
                jNy_t0, jNz_t0)],
                ...,
                [np.array([j0x_tN, j0y_tN, j0z_tN]),...,np.array(jNx_tN,
                jNy_tN, jNz_tN)]])



        PARAMETERS
        ----------
        save_location: string
            the location in the database pointing to the recorded data
            for arm tests usually in the form of
            'test_group/test_name/session%03d/run%03d/'
        interpolated_samples: positive int, Optional (Default=100)
            the number of samples to take (evenly) from the interpolated data
            if set to None, no interpolated or sampling will be done, the raw
            data will be returned
        clear_memory: boolean, Optional (Default: True)
            True: overwrite the instantiated objects to None to save memory if
            looping through runs or tests, instead of keeping several
            instantiations of the database, dataprocessor, and robot_config objects
            False: leave the database, dataprocessor, and robot_config objects
            in case they need to be referenced.
        """
        # create a dictionary with our test data interpolated to the same
        # number of steps
        data = self.load_and_process(save_location=save_location,
                interpolated_samples=interpolated_samples)

        if self.robot_config is not None:
            # get the cartesian coordinates of the virtual arm joints and links
            [data['joints_xyz'], data['links_xyz']] = self.calc_cartesian_points(
                robot_config=self.robot_config, q=data['q'])

            #TODO: need to implement limit checking - basic will be checking
            # collision with ground
            # # check if any of the cartesian points fall passed our working limit
            # self.check_limits()

        if clear_memory:
            self.robot_config = None
            self.dat = None
            self.proc = None

        return data
