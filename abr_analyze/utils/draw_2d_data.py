from abr_analyze.utils.data_visualizer import DataVisualizer
from abr_analyze.utils.data_processor import DataProcessor
from .draw_data import DrawData
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Draw2dData(DrawData):
    '''
        A class for plotting database parameters onto a 2d ax object
    '''
    def __init__(self, db_name, interpolated_samples=100):
        '''
            PARAMETERS
            ----------
            db_name: string
                the name of the database to load
            interpolated_samples: positive int, Optional (Default=100)
                the number of samples to take (evenly) from the interpolated data
                if set to None, no interpolated or sampling will be done, the raw
                data will be returned, Use None for no interpolation
        '''
        super(Draw2dData, self).__init__()
        self.db_name = db_name
        self.interpolated_samples = interpolated_samples
        # create a dict to store processed data
        self.data = {}
        # instantiate our process and visualize modules
        self.proc = DataProcessor()
        self.vis = DataVisualizer()

    def plot(self, ax, save_location, parameters, step=-1, c=None, linestyle='--'):
        '''
            Plots the parameters from save_location on the ax object
            Returns the ax object and the current max x and y limits

            PARAMETERS
            ----------
            ax: ax object for plotting
            save_location: string
                points to the location in the hdf5 database to read from
            parameters: string or list of strings
                The parameters to load from the save location, can be a single
                parameter, or a list of parameters
            step: int, Optional (Default: -1)
                the position in the data list to plot to, when -1 is used the
                entire dataset will be plotted
            c: string, Optional (Default: None)
                matplotlib compatible color to be used when plotting data
            linestyle: string, Optional (Default: None)
                matplotlib compatible linestyle to be used when plotting data
        '''
        # convert our parameters to lists if they are not
        parameters = self.make_list(parameters)
        ax = self.make_list(ax)

        # check if the specific save location and parameter set have been
        # processed already to avoid reprocessing if plotting the same data at
        # a different step or onto a different axis
        save_name = '%s-%s'%(save_location, parameters)
        if save_name not in self.data:
            self.data[save_name] = self.proc.load_and_process(db_name=self.db_name,
                    save_location=save_location, parameters=parameters,
                    interpolated_samples=self.interpolated_samples)

        for param in parameters:
            # remove single dimensions
            self.data[save_name][param] = np.squeeze(self.data[save_name][param])
            # avoid passing time in for finding y limits
            if param is not 'time':
                # update our x and y limits with every test we add
                self.check_plot_limits(x=np.cumsum(self.data[save_name]['time']),
                        y=self.data[save_name][param])

            ax = self.vis.plot_2d_data(ax=ax, x=np.cumsum(self.data[save_name]['time'])[:step],
                    y=self.data[save_name][param][:step], c=c,
                    linestyle=linestyle)

        # ax.set_xlim(self.xlimit[0], self.xlimit[1])
        # ax.set_ylim(self.ylimit[0], self.ylimit[1])
        return [ax, [self.xlimit, self.ylimit]]
