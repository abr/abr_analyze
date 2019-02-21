from abr_analyze.utils.data_visualizer import DataVisualizer
from abr_analyze.utils.data_processor import DataProcessor
from .draw_data import DrawData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
"""
"""
class DrawArm(DrawData):
    '''
        A class for plotting a stick arm onto a 3d ax object
    '''
    def __init__(self, db_name, robot_config, interpolated_samples=100,
                 show_trajectory=True):
        '''
            PARAMETERS
            ----------
            db_name: string
                the name of the database to load
            robot_config: object
                an instantiated abr_control arm config
            interpolated_samples: positive int, Optional (Default=100)
                the number of samples to take (evenly) from the interpolated data
                if set to None, no interpolated or sampling will be done, the raw
                data will be returned, Use None for no interpolation
            show_trajectory: boolean, Optional (Default: True)
                True to show the end-effector outline

        '''
        super(DrawArm, self).__init__()

        self.projection='3d'
        self.db_name = db_name
        self.robot_config = robot_config
        self.interpolated_samples = interpolated_samples
        self.show_trajectory = show_trajectory
        # create a dict to store processed data
        self.data = {}
        # instantiate our process and visualize modules
        self.vis = DataVisualizer()
        self.proc = DataProcessor()

    def plot(self, ax, save_location, step=-1, parameters=None, c='b', linestyle=None,
            label=None, title=None):
        '''
            Plots the parameters from save_location on the ax object
            Returns the ax object and the current max x, y and z limits

            PARAMETERS
            ----------
            ax: ax object for plotting
            save_location: string
                points to the location in the hdf5 database to read from
            parameters: None
                This is not used in this function, but has been keep consistent
                with the class
            step: int, Optional (Default: -1)
                the position in the data list to plot to, when -1 is used the
                entire dataset will be plotted
            c: string, Optional (Default: None)
                matplotlib compatible color to be used when plotting data
            linestyle: string, Optional (Default: None)
                matplotlib compatible linestyle to be used when plotting data

        '''
        if save_location not in self.data:
            # create a dictionary with our test data interpolated to the same
            # number of steps
            self.data[save_location] = self.proc.load_and_process(db_name=self.db_name,
                    save_location=save_location,
                    interpolated_samples=self.interpolated_samples,
                    parameters=['time', 'q'])
            # get our joint and link positions
            [joints, links, ee_xyz] = self.proc.calc_cartesian_points(
                    robot_config=self.robot_config,
                    q=self.data[save_location]['q'])
            self.data[save_location]['joints_xyz'] = joints
            self.data[save_location]['links_xyz'] = links
            self.data[save_location]['ee_xyz'] = ee_xyz

        data = self.data[save_location]

        # plot our arm figure
        self.vis.plot_arm(ax=ax, joints_xyz=data['joints_xyz'][step],
                links_xyz=data['links_xyz'][step], ee_xyz=data['ee_xyz'][step],
                title=title)

        # plot the ee trajectory
        if self.show_trajectory:
            self.vis.plot_3d_data(ax=ax, data=data['ee_xyz'][:step], c=c,
                    linestyle=linestyle, label=label)

        # ax.set_title(save_location)
        return [ax, [self.xlimit, self.ylimit, self.zlimit]]
