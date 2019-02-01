#TODO: make this plot only a single ax object with parameters to either pass an
# ax object, if not one is created since we only want the one frame, otherwise
# get the grid layout done in a higher level script
import abr_jaco2
from abr_analyze.utils.data_visualizer import DataVisualizer
from abr_analyze.utils.data_processor import DataProcessor

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import os
"""
"""
class DrawTrajectory():
    '''

    '''
    def __init__(self, db_name, interpolated_samples=100):
        '''

        '''
        self.db_name = db_name
        self.interpolated_samples = interpolated_samples
        # create a dict to store processed data
        self.data = {}
        # instantiate our process and visualize modules
        self.proc = DataProcessor()
        self.vis = DataVisualizer()

    def plot(self, ax, save_location, step, param, c='tab:purple', linestyle='--'):
        '''

        '''
        save_name = '%s-%s'%(save_location, param)
        if save_name not in self.data:
            self.data[save_name] = self.proc.load_and_process(db_name=self.db_name,
                    save_location=save_location, params=[param],
                    interpolated_samples=self.interpolated_samples)

        data = self.data[save_name]

        self.vis.plot_trajectory(ax=ax, data=data[param][:step], c=c,
                linestyle=linestyle)

        return ax
