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
class Draw3dData():
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
        # set our plot limits to zero and overwrite as we pass data in
        self.xlimit = [0,0]
        self.ylimit = [0,0]
        self.zlimit = [0,0]

    def check_xyz_limits(self, x, y, z):
        if x.ndim > 1:
            self.xlimit[0] = min(min(x.min(axis=1)), self.xlimit[0])
            self.xlimit[1] = max(max(x.max(axis=1)), self.xlimit[1])
        else:
            self.xlimit[0] = min(min(x), self.xlimit[0])
            self.xlimit[1] = max(max(x), self.xlimit[1])

        if y.ndim > 1:
            self.ylimit[0] = min(min(y.min(axis=1)), self.ylimit[0])
            self.ylimit[1] = max(max(y.max(axis=1)), self.ylimit[1])
        else:
            self.ylimit[0] = min(min(y), self.ylimit[0])
            self.ylimit[1] = max(max(y), self.ylimit[1])

        if z.ndim > 1:
            self.zlimit[0] = min(min(z.min(axis=1)), self.zlimit[0])
            self.zlimit[1] = max(max(z.max(axis=1)), self.zlimit[1])
        else:
            self.zlimit[0] = min(min(z), self.zlimit[0])
            self.zlimit[1] = max(max(z), self.zlimit[1])

    def plot(self, ax, save_location, step, param, c='tab:purple', linestyle='--'):
        '''

        '''
        for location in save_location:
            save_name = '%s-%s'%(location, param)
            if save_name not in self.data:
                self.data[save_name] = self.proc.load_and_process(db_name=self.db_name,
                        save_location=location, params=[param],
                        interpolated_samples=self.interpolated_samples)

                data = self.data[save_name]

                # update our xyz limit with every test we add
                self.check_xyz_limits(
                        x=data[param][:,0],
                        y=data[param][:,1],
                        z=data[param][:,2])

                self.vis.plot_trajectory(ax=ax, data=data[param][:step], c=c,
                        linestyle=linestyle)

        ax.set_xlim(self.xlimit[0], self.xlimit[1])
        ax.set_ylim(self.ylimit[0], self.ylimit[1])
        ax.set_zlim(self.zlimit[0], self.zlimit[1])

        return ax
