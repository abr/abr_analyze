#TODO: make this plot only a single ax object with parameters to either pass an
# ax object, if not one is created since we only want the one frame, otherwise
# get the grid layout done in a higher level script
import abr_jaco2
from abr_analyze.utils.data_visualizer import DataVisualizer
from abr_analyze.utils.data_processor import DataProcessor
from .draw_data import DrawData

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import os
"""
"""
class DrawArm(DrawData):
    '''

    '''
    def __init__(self, db_name, robot_config=None, interpolated_samples=100,
            show_filter=True, show_trajectory=True):
        '''

        '''
        super(DrawArm, self).__init__()

        self.db_name = db_name
        self.robot_config = robot_config
        self.interpolated_samples = interpolated_samples
        self.show_filter = show_filter
        self.show_trajectory = show_trajectory
        # create a dict to store processed data
        self.data = {}
        # instantiate our process and visualize modules
        self.vis = DataVisualizer()
        self.proc = DataProcessor()

    def plot(self, ax, save_location, step, param=None, c='b', linestyle=None):
        '''

        '''
        if not isinstance(save_location, list):
            save_location = [save_location]

        for location in save_location:
            save_name = location
            if save_name not in self.data:
                # create a dictionary with our test data interpolated to the same
                # number of steps
                self.data[save_name] = self.proc.load_and_process(db_name=self.db_name,
                        save_location=location,
                        interpolated_samples=self.interpolated_samples,
                        params=['ee_xyz', 'target', 'time', 'filter', 'q'])
                # get our joint and link positions
                [joints, links] = self.proc.calc_cartesian_points(
                        robot_config=self.robot_config,
                        q=self.data[save_name]['q'])
                self.data[save_name]['joints_xyz'] = joints
                self.data[save_name]['links_xyz'] = links

            data = self.data[save_name]

            # plot our arm figure
            self.vis.plot_arm(ax=ax, joints_xyz=data['joints_xyz'][step],
                    links_xyz=data['links_xyz'][step], ee_xyz=data['ee_xyz'][step])

            # plot the filtered target trajectory
            if self.show_filter:
                self.vis.plot_trajectory(ax=ax, data=data['filter'][:step], c='g',
                        linestyle='-')

            # plot the ee trajectory
            if self.show_trajectory:
                self.vis.plot_trajectory(ax=ax, data=data['ee_xyz'][:step], c=c,
                        linestyle=linestyle)

        ax.set_title(save_location)
        ax.set_xlim3d(-0.5,0.5)
        ax.set_ylim3d(-0.5,0.5)
        ax.set_zlim3d(0,1)
        ax.set_aspect(1)

        return ax
