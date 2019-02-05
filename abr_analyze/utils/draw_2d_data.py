#TODO: make this plot only a single ax object with parameters to either pass an
# ax object, if not one is created since we only want the one frame, otherwise
# get the grid layout done in a higher level script
import abr_jaco2
from abr_analyze.utils.data_visualizer import DataVisualizer
from abr_analyze.utils.data_processor import DataProcessor
from .draw_data import DrawData

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import os
"""
"""
class Draw2dData(DrawData):
    '''

    '''
    def __init__(self, db_name, interpolated_samples=100):
        '''

        '''
        super(Draw2dData, self).__init__()

        self.db_name = db_name
        self.interpolated_samples = interpolated_samples
        # create a dict to store processed data
        self.data = {}
        # instantiate our process and visualize modules
        self.proc = DataProcessor()
        self.vis = DataVisualizer()

    def plot(self, ax, save_location, step, param, c=None, linestyle='--'):
        '''

        '''
        if not isinstance(save_location, list):
            save_location = [save_location]

        for location in save_location:
            save_name = '%s-%s'%(location, param)
            if save_name not in self.data:
                self.data[save_name] = self.proc.load_and_process(db_name=self.db_name,
                        save_location=location, params=[param],
                        interpolated_samples=self.interpolated_samples)

            if self.data[save_name][param].ndim == 2:
                self.data[save_name][param] = np.squeeze(self.data[save_name][param])

            data = self.data[save_name]

            # update our x and y limits with every test we add
            self.check_plot_limits(x=np.cumsum(data['time']), y=data[param])

            self.vis.plot_data(ax=ax, x=np.cumsum(data['time'])[:step], y=data[param][:step], c=c,
                    linestyle=linestyle)

        ax.set_xlim(self.xlimit[0], self.xlimit[1])
        ax.set_ylim(self.ylimit[0], self.ylimit[1])
        return ax
