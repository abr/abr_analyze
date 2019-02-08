import numpy as np
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import axes3d
from abr_analyze.utils.paths import figures_dir
import uuid

class DrawCells():
    def __init__(self):
        self.data = {'cell_ids': []}
        self.animate_steps = 1

    def cell_to_subplot(self, cell, n_rows, n_cols,  projection=None):
        '''
        Accepts a gridspec.Gridspec(n,m)[x,y] cell, and breaks that location
        up into n_rows and n_cols ax objects, returns a list of ax objects

        *EXAMPLE*
        outer_grid = gridspec.GridSpec(3,3)
        cell = outer_grid[1,2]
        ax = cell_to_subplot(cell=cell, n_rows=3, n_cols=1)
        ax[0].plot(data0)
        ax[1].plot(data1)
        ax[2].plot(data2)
        plt.show()
        '''
        inner_grid = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, cell)
        ax = []
        for row in range(0, n_rows):
            for col in range(0, n_cols):
                if projection is None:
                    ax.append(plt.subplot(inner_grid[row,col]))
                else:
                    ax.append(plt.subplot(inner_grid[row,col],
                            projection=projection))
                # self.master_ax.append(plt.subplot(inner_grid[row,col]))
        return ax

    def add_cell(self, cell, function, save_location, parameters='None', n_rows=1, n_cols=1,
            animate=False):

        # get the memory location of the cell so we don't reprocess
        cell_id = hex(id(cell))
        # create a unique id to link the parameter set to the cell id
        param_id = uuid.uuid4()
        # save the parameters to a dict for later plotting
        param_dict = {'function': function, 'save_location': save_location,
                'parameters': parameters, 'animate': animate}

        # if this is the first time the cell is passed in, convert it to the
        # specified number of row and column ax objects
        if cell_id not in self.data['cell_ids']:
            # get the number of interpolated samples for the function to find
            # the maximum value to loop through when animating
            if isinstance(function.interpolated_samples, int) and animate:
                self.animate_steps = max(function.interpolated_samples,
                        self.animate_steps)
            self.data['cell_ids'].append(cell_id)
            axes = self.cell_to_subplot(cell=cell, n_rows=n_rows,
                    n_cols=n_cols, projection=function.projection)
            # save the ax objects and parameters linked to this cell
            cell_dict = {'ax': axes, 'param_ids': [param_id], param_id: param_dict}
            self.data[cell_id] = cell_dict
        # this is the second time we receive the same cell object, append the
        # parameter dict to it for multiple plotting on the same axes
        else:
            self.data[cell_id][param_id] = param_dict
            self.data[cell_id]['param_ids'].append(param_id)

    def generate(self):
        cell_ids = self.data['cell_ids']
        # this will only be greater than one if the cell is being animated
        if self.animate_steps > 1:
            from abr_analyze.utils import MakeGif
            gif = MakeGif()
            fig_cache = gif.prep_fig_cache()

        for ii in range(0, self.animate_steps):
            # TODO: add progress printout
            # loop through each cell in the plot
            for cell_id in cell_ids:
                # get the ax object(s) and parameter sets linked to this cell
                cell_data = self.data[cell_id]
                ax = cell_data['ax']
                # get all the parameter sets to be plotted in this cell
                param_ids = cell_data['param_ids']
                for param_id in param_ids:
                    data = cell_data[param_id]
                    # if this parameter set is to be animated, pass in the
                    # current step, else plot the entire dataset
                    if data['animate'] is False:
                        step = -1
                    # if for some stupid reason you decide to have different
                    # interpolated_samples values for different tests, this
                    # will catch that and plot to the end of the data set for
                    # parameter sets with interpolated_samples < ii
                    elif data['function'].interpolated_samples < ii:
                        step = -1
                    else:
                        step = ii

                    [ax,limits] = data['function'].plot(
                            ax=ax,
                            save_location=data['save_location'],
                            step=step,
                            parameters=data['parameters']
                            )
                    #TODO: link the x and y limits to the axis
            #TODO: set axis limits
            if self.animate_steps > 1:
                save_loc = '%s/%05d.png'%(fig_cache, ii)
                plt.savefig(save_loc)
                for cell_id in cell_ids:
                    # get the ax object(s) and parameter sets linked to this cell
                    cell_data = self.data[cell_id]
                    ax = cell_data['ax']
                    for a in ax:
                        a.clear()
            else:
                plt.show()
            #TODO: clear all ax's here
        if self.animate_steps > 1:
            save_loc='%s/draw_cells'%(figures_dir)
            gif.create(fig_loc=fig_cache,
                       save_loc=save_loc,
                       save_name='draw_cells_test',
                       delay=5, res=[1920,1080])