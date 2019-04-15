"""
A visualizer for looking through intercept scan data. Currently the test group
save name has to be manually changed at the bottom of the script.
"""
import tkinter as tk
from tkinter import ttk
import time

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style
import seaborn
from PIL import Image, ImageTk

import numpy as np
from abr_analyze.data_handler import DataHandler
from abr_analyze.paths import figures_dir

from terminaltables import AsciiTable

# a reorganized dict of intercept values and run number to allow for easy
# searching. Allows the user to find the desired run number by searching
# key_dict[left_bound][right_bound][mode]
global key_dict
# the current intercept values in the button selection windows
global intercept_vals
# the boolean value of the plot buttons
global save_val
global keep_test_val
global toggle_ideal_val
global clear_plot_val
global legend_loc_val
global update_plot
# plot colors
global plt_col

class FontsAndColors():
    '''
    constants for text styles and colors
    '''
    def __init__(self):
        # colors for terminal tables
        self.HEADER = '\033[95m'
        self.BLUE = '\033[94m'
        self.GREEN = '\033[92m'
        self.YELLOW= '\033[93m'
        self.RED = '\033[91m'
        self.ENDC = '\033[0m'
        self.BOLD = '\033[1m'
        self.UNDERLINE = '\033[4m'
        # set some constants
        self.XL_FONT = ("Verdana", 50)
        self.LARGE_FONT = ("Verdana", 20)
        self.MED_FONT = ("Verdana", 10)
        self.SMALL_FONT = ("Verdana", 8)
        style.use("ggplot")
        self.button_text_color = 'white'
        self.text_color = '#36454f'
        # charcoal grey
        # background_color = '#36454f'
        self.background_color = 'white'
        # abr blue
        self.button_color = '#375580'
        self.button_text_color = '#d3d3d3'#'white'
        self.secondary_text = 'black'
        self.secondary_bg = '#ffff7f'

class ButtonFun():
    '''
    a class for all of the button functions
    '''
    def val_up(self, val, step):
        '''
        increases val by step

        PARAMETERS
        ----------
        val: tk.StringVar()
            a value linked to a button to increment
        step: float
            the amount to increase val by
        '''
        global update_plot
        val_num = float(val.get())
        val_num += step
        val.set('%.1f'%val_num)
        update_plot = True

    def val_down(self, val, step):
        '''
        decreases val by step

        PARAMETERS
        ----------
        val: tk.StringVar()
            a value linked to a button to decrement
        step: float
            the amount to decrease val by
        '''
        global update_plot
        val_num = float(val.get())
        val_num -= step
        val.set('%.1f'%val_num)
        update_plot = True

    def save(*args):
        '''
        updates the global value that tells the gui to save the current figure
        '''
        global update_plot
        global save_val
        save_val = True
        update_plot = True

    def keep_test(*args):
        '''
        updates the global value that tells the gui to keep the current test
        plotted on the figure
        '''
        global update_plot
        global keep_test_val
        keep_test_val = True
        update_plot = True

    def toggle_ideal(*args):
        '''
        updates the global value that tells the gui to toggle the current
        status of the ideal baseline between visible and not
        '''
        global update_plot
        global toggle_ideal_val
        toggle_ideal_val = not toggle_ideal_val
        update_plot = True

    def clear_plot(*args):
        '''
        updates the global value that tells the gui to clear the figure
        '''
        global update_plot
        global clear_plot_val
        clear_plot_val = True
        update_plot = True

    def legend_loc(*args):
        '''
        updates the global value that tells the gui to cycle to the next legend
        location
        '''
        global update_plot
        global legend_loc_val
        legend_loc_val += 1
        update_plot = True

class GuiItems():
    '''
    Class for creating common gui objects
    '''
    def __init__(self):
        self.pars = FontsAndColors()
    def button(self, frame, text, function, row=1, col=1,
            f_col=None, b_col=None):
        '''
        creates a button in the gui

        PARAMETERS
        ----------
        frame: tk.Frame() object
        text: string
            label for the button
        function: function
            the function the button will trigger
        row: int, Optional (Default: 1)
            the row location of the button
        col: int, Optional (Default: 1)
            the col location of the button
        f_col: string, Optional (Default: None)
            the color of the button foreground
        b_col: string, Optional (Default: None)
            the color of the button background
        '''
        if f_col is None:
            f_col = self.pars.button_text_color
        if b_col is None:
            b_col = self.pars.button_color
        butn = tk.Button(frame, text=text,
                command=function)
        butn.grid(row=row, column=col, sticky='nsew')
        butn.configure(background=b_col, foreground=f_col)
        return np.copy(butn)

    def entry_box(self, frame, text_var, width=1, row=1, col=1,
            f_col=None, b_col=None, rowspan=1, colspan=1):
        '''
        creates an entry box in the gui

        PARAMETERS
        ----------
        frame: tk.Frame() object
        text_var: tk.StringVar()
            the string variable to link to the entry box, this is the variable
            that will be updated with the value in the entry box
        width: int, Optional (Default: 1)
            the width of the entry box
        row: int, Optional (Default: 1)
            the row location of the button
        col: int, Optional (Default: 1)
            the col location of the button
        f_col: string, Optional (Default: None)
            the color of the button foreground
        b_col: string, Optional (Default: None)
            the color of the button background
        rowspan: int, Optional (Default: 1)
            the amount of point in the grid to span across in x
        colspan: int, Optional (Default: 1)
            the amount of point in the grid to span across in y
        '''
        if f_col is None:
            f_col = self.pars.text_color
        if b_col is None:
            b_col = self.pars.background_color
        entry = tk.Entry(frame, textvariable=text_var, width=width)
        entry.grid(row=row, column=col, columnspan=colspan,
                rowspan=rowspan, sticky='nsew')
        entry.configure(background=b_col, foreground=f_col)
        return np.copy(entry)

    def label(self, frame, textvariable=None, text=None,
            row=1, col=1, font=None, height=0, width=0,
            f_col=None, b_col=None, rowspan=1, colspan=1):
        '''
        creates a label in the gui

        PARAMETERS
        ----------
        frame: tk.Frame() object
        textvariable: tk.StringVar(), Optional (Default: None)
            the string variable to link to the entry box, this is the variable
            that will be updated with the value in the entry box
        text: string, Optional (Default: None)
            the string that will be placed in the label
        row: int, Optional (Default: 1)
            the row location of the button
        col: int, Optional (Default: 1)
            the col location of the button
        font: string, Optional (Default, None)
            the font for the label
        width: int, Optional (Default: 0)
            the width of the entry box
        height: int, Optional (Default: 0)
            the height of the entry box
        f_col: string, Optional (Default: None)
            the color of the button foreground
        b_col: string, Optional (Default: None)
            the color of the button background
        rowspan: int, Optional (Default: 1)
            the amount of point in the grid to span across in x
        colspan: int, Optional (Default: 1)
            the amount of point in the grid to span across in y

        NOTE* the label can either be simple text, or a tk.StringVar() that
        will change the value of the label. If both are passed in the simple
        string will be used
        '''
        if f_col is None:
            f_col = self.pars.text_color
        # elif type(f_col) is not str:
        #     f_col = f_col.get()

        if b_col is None:
            b_col = self.pars.background_color
        # elif type(b_col) is not str:
        #     b_col = b_col.get()

        if font is None:
            font = self.pars.MED_FONT
        # elif type(font) is not str:
        #     font = font.get()

        if text is not None:
            labl = tk.Label(frame, text=text, font=font, height=height,
                    width=width)
        elif textvariable is not None:
            labl = tk.Label(frame, textvariable=textvariable, font=font)
        else:
            raise Exception ('Either text or textvariable needs to be passed in')
        labl.grid(row=row, column=col, columnspan=colspan, rowspan=rowspan,
                sticky='nsew')
        labl.configure(background=b_col, foreground=f_col)
        return np.copy(labl)

class MasterPage(tk.Tk):
    '''
    The master page container that is created to be filled with a grid of
    tk.Frame() objects
    '''
    def __init__(self, db_name, save_location, *args, **kwargs):
        '''
        PARAMETERS
        ----------
        db_name: string
            the name of the database that holds the intercept scans
        save_location: string
            the location in the database the data is saved
        '''

        # instantiate our container
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, 'abr_analyze\n Intercept Scan Viewer')

        container = tk.Frame(self)
        #container.configure(background='white')
        container.grid(row=1, column=1, sticky='nsew')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.frames = {}

        # place our other pages into the master page
        frame = InterceptScanViewer(parent=container, db_name=db_name,
                save_location=save_location, *args)
        self.frames[InterceptScanViewer] = frame
        frame.grid(row=2, column=2, sticky="nsew")
        self.show_frame(page_container=InterceptScanViewer)

    def show_frame(self, page_container):
        """
        Function for pulling the selected frame to the front
        """
        frame = self.frames[page_container]
        frame.tkraise()

class InterceptScanViewer(tk.Frame):
    '''
    the intercept scanner page
    '''
    def __init__(self, parent, db_name, save_location, *args):
        '''
        PARAMETERS
        ----------
        db_name: string
            the name of the database that holds the intercept scans
        save_location: string
            the location in the database the data is saved
        '''
        global key_dict
        global intercept_vals
        global save_val
        global keep_test_val
        global toggle_ideal_val
        global clear_plot_val
        global legend_loc_val
        global update_plot
        global plt_col

        save_val = False
        keep_test_val = False
        toggle_ideal_val = True
        clear_plot_val = False
        legend_loc_val = 1
        update_plot = True
        # set some step boundaries for possible values
        mode_step = 0.1
        bound_step = 0.1
        mode_range = [-0.9, 0.9]
        bound_range = [-0.9, 0.9]
        plt_col = ['r', 'b', 'y', 'k', 'tab:grey']

        # instanitate our item creating class
        self.create = GuiItems()

        # instantiate our gui parameter class
        self.pars = FontsAndColors()

        # instantiate our button function class
        self.button = ButtonFun()

        # instantiate our data loading class and get defaults
        self.dat = DataHandler(db_name=db_name)
        data = self.dat.load(
                parameters=['ideal', 'total_intercepts'],
                save_location=save_location)
        self.ideal = data['ideal']
        runs = data['total_intercepts']

        key_dict = {}
        for ii in range(0,runs):
            data = self.dat.load(
                    parameters=['intercept_bounds','intercept_mode'],
                    save_location='%s/%05d'%(save_location,ii))
            left_bound = '%.1f'%data['intercept_bounds'][0]
            right_bound = '%.1f'%data['intercept_bounds'][1]
            mode = '%.1f'%data['intercept_mode']

            # if first time with this left bound, save all the data now
            if left_bound not in key_dict:
                key_dict[left_bound] = {right_bound: {mode: '%05d'%ii}}
            # left bound exists, check if the right bound has already been saved
            elif right_bound not in key_dict[left_bound]:
                key_dict[left_bound][right_bound] = {mode: '%05d'%ii}
            # left and right bound combination already exist, check if mode saved
            elif mode not in key_dict[left_bound][right_bound]:
                key_dict[left_bound][right_bound][mode] = '%05d'%ii

        # instantiate our frame
        tk.Frame.__init__(self, parent)
        self.configure(background=self.pars.background_color)

        # instantiate the cells in our main frame
        # CELL 1: cell for our plot
        frame_plot = tk.Frame(self, parent)
        frame_plot.grid(row=0,column=0, padx=10)
        frame_plot.configure(background=self.pars.background_color)

        # CELL 2: cell for our save / hold / clear buttons
        frame_plot_buttons = tk.Frame(self, parent)
        frame_plot_buttons.grid(row=0, column=1, padx=10)
        frame_plot.configure(background=self.pars.background_color)

        # CELL 3: cell for our intercept setting buttons and text boxes
        frame_intercept_val = tk.Frame(self, parent)
        frame_intercept_val.grid(row=1, column=0, padx=10)
        frame_intercept_val.configure(background=self.pars.background_color)

        # CELL 4: frame for printouts / notes to user
        frame_notes = tk.Frame(self, parent)
        frame_notes.grid(row=1, column=1, padx=10)
        frame_notes.configure(background=self.pars.background_color)

        # create our plotting buttons
        keep_button = self.create.button(
            frame=frame_plot_buttons,
            text='Keep Test',
            function=lambda: self.button.keep_test(self),
            row=0, col=0)

        clear_button = self.create.button(
            frame=frame_plot_buttons,
            text='Clear Plot',
            function=lambda: self.button.clear_plot(self),
            row=1, col=0)

        # ideal_button = self.create.button(
        #     frame=frame_plot_buttons,
        #     text='Toggle Ideal',
        #     function=lambda: self.button.toggle_ideal(self),
        #     row=2, col=0)

        save_button = self.create.button(
            frame=frame_plot_buttons,
            text='Save',
            function=lambda: self.button.save(self),
            row=3, col=0)

        legend_loc_button = self.create.button(
            frame=frame_plot_buttons,
            text='Legend Loc',
            function=lambda: self.button.legend_loc(self),
            row=4, col=0)

        # def callback(*args):
        #     print('button press')
        # create our string variables for intercept values
        left_bound_val = tk.StringVar()
        #left_bound_val.trace('w', callback)
        left_bound_val.set(left_bound)
        right_bound_val = tk.StringVar()
        #right_bound_val.trace('w', callback)
        right_bound_val.set(right_bound)
        mode_val = tk.StringVar()
        #mode_val.trace('w', callback)
        mode_val.set(mode)

        # set our intercept_vals to a starting value
        intercept_vals = [left_bound_val, right_bound_val, mode_val]


        # create our intercept setting buttons
        left_bound_up = self.create.button(
            frame=frame_intercept_val,
            text='/\\',
            function=lambda: self.button.val_up(left_bound_val, bound_step),
            row=0, col=0)

        left_bound_down = self.create.button(
            frame=frame_intercept_val,
            text='\/',
            function=lambda: self.button.val_down(left_bound_val, bound_step),
            row=1, col=0)

        right_bound_up = self.create.button(
            frame=frame_intercept_val,
            text='/\\',
            function=lambda: self.button.val_up(right_bound_val, bound_step),
            row=0, col=4)

        right_bound_down = self.create.button(
            frame=frame_intercept_val,
            text='\/',
            function=lambda: self.button.val_down(right_bound_val, bound_step),
            row=1, col=4)

        mode_up = self.create.button(
            frame=frame_intercept_val,
            text='/\\',
            function=lambda: self.button.val_up(mode_val, mode_step),
            row=0, col=2)

        mode_down = self.create.button(
            frame=frame_intercept_val,
            text='\/',
            function=lambda: self.button.val_down(mode_val, mode_step),
            row=1, col=2)

        # create our intercept setting entry box
        left_bound_entry = self.create.entry_box(
                frame=frame_intercept_val,
                text_var=left_bound_val,
                row=0, col=1)

        right_bound_entry = self.create.entry_box(
                frame=frame_intercept_val,
                text_var=right_bound_val,
                row=0, col=5)

        mode_entry = self.create.entry_box(
                frame=frame_intercept_val,
                text_var=mode_val,
                row=0, col=3)

        # labels for entry boxes
        left_bound_label = self.create.label(
                frame=frame_intercept_val,
                text='Left',
                row=1, col=1)

        right_bound_label = self.create.label(
                frame=frame_intercept_val,
                text='Right',
                row=1, col=5)

        mode_label = self.create.label(
                frame=frame_intercept_val,
                text='Mode',
                row=1, col=3)

        # label that triggers if test with selected values does not exist
        global valid_val
        #global valid_col
        valid_val = tk.StringVar()
        #valid_col = tk.StringVar()
        valid_val.set('')
        #valid_col.set('green')
        valid_label = self.create.label(
                frame=frame_notes,
                textvariable=valid_val,
                row=1, col=1,
                font=self.pars.XL_FONT)

        # Plotting Window
        canvas = FigureCanvasTkAgg(live.fig, self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0)
        canvas.get_tk_widget().configure(background=self.pars.background_color)

class LiveFigure():
    '''
    Class that handles the live plotting of the figure in the gui
    '''
    def __init__(self, db_name, save_location):

        '''
        PARAMETERS
        ----------
        db_name: string
            the name of the database that holds the intercept scans
        save_location: string
            the location in the database the data is saved
        '''
        self.save_location = save_location
        self.dat = DataHandler(db_name)
        self.fig = Figure()#figsize=(10,12), dpi=100)
        self.a = self.fig.add_subplot(111)
        self.ideal = self.dat.load(
                parameters=['ideal'],
                save_location=self.save_location)['ideal']
        self.test_que = []

    def plot(self,i):
        '''
        plots the data specified in the global variables that get updated with
        the functions linked to the gui buttons
        '''
        global save_val
        global keep_test_val
        global toggle_ideal_val
        global clear_plot_val
        global legend_loc_val
        global update_plot
        global plt_col

        if update_plot:
            update_plot = False
            if clear_plot_val:
                data = None
                self.test_que = []
                a = self.fig.add_subplot(111)
                a.clear()
                clear_plot_val = False
            else:
                intercept_keys = self.get_intercept_vals_from_buttons()
                data = self.load(keys=intercept_keys)

                if data is not None:
                    a = self.fig.add_subplot(111)
                    a.clear()
                    label = '(%.1f, %.1f), %.1f\nACTV:%i | INACTV:%i | %.2f'%(
                        data['intercept_bounds'][0],
                        data['intercept_bounds'][1],
                        data['intercept_mode'],
                        data['num_active'],
                        data['num_inactive'],
                        data['y_mean']
                        )
                    if data['title'] == 'proportion_time_neurons_active':
                        a.set_xlim(0, 1)
                        a.bar(data['x'], data['y'], width=1/(2*self.bins),
                              label=label, edgecolor='white',
                              alpha=0.5, color=plt_col[0])
                    else:
                        a.set_ylim(0, 1)
                        a.plot(data['x'], data['y'], label=label, c=plt_col[0])
                    a.set_title(data['title'])

                    if keep_test_val:
                        current_test = {'label': label, 'y': data['y'],
                                'x': data['x']}
                        self.test_que.append(current_test)
                        keep_test_val = False

                    if self.test_que:
                        for aa, test in enumerate(self.test_que):
                            if aa+1 >= len(plt_col):
                                plt_col.append(None)
                            if data['title'] == 'proportion_time_neurons_active':
                                a.bar(test['x'], test['y'],
                                      width=1/(2*self.bins),
                                      label=test['label'],
                                      edgecolor='white', alpha=0.5,
                                      color=plt_col[aa+1])
                            else:
                                a.plot(test['x'], test['y'],
                                        label=test['label'], c=plt_col[aa+1])

                    # if toggle_ideal_val:
                    #     a.plot(data['x'], self.ideal, label='ideal',
                    #             c='k', lw=3)
                    a.legend(loc=legend_loc_val%4+1)

                    if save_val:
                        #global valid_val
                        a.figure.savefig('%s/intercept_scan_%s.png'
                                         % (figures_dir, data['title']))
                        msg = ('Figure saved to:'
                                + ' %s/intercept_scan_%s.png'
                                % (figures_dir, data['title']))
                        print(msg)
                        #TODO: make the font smaller while this is printed out
                        #valid_val.set(msg)
                        #time.sleep(1)
                        save_val = False


    def get_intercept_vals_from_buttons(self):
        '''
        updates the intercept values that are linked to a specific test based
        on the values in the text boxes
        '''
        global intercept_vals
        intercept_keys = []
        for ii, val in enumerate(intercept_vals):
            intercept_keys.append(val.get())
        return intercept_keys

    def load(self, keys):
        '''
        loads the required data for plotting from the test specified by the
        provided keys

        PARAMETERS
        ----------
        keys: list of 3 strings
            this list is created in the intercept scanner page and it links the
            test numbers to their intercept ranges and mode
        '''
        global key_dict
        global valid_val
        self.bins = 30
        try:
            test_num = int(key_dict[keys[0]][keys[1]][keys[2]])
            data = self.dat.load(
                    parameters=['intercept_bounds', 'intercept_mode', 'y',
                        'title', 'num_active', 'num_inactive'],
                    save_location='%s/%05d'%(self.save_location, test_num))
            # get the mean before converting y to a histogram
            data['y_mean'] = np.mean(data['y'])
            if data['title'] == 'proportion_time_neurons_active':
                y, bins_out = np.histogram(np.squeeze(data['y']),
                                           bins=np.linspace(0, 1, self.bins))
                data['x'] = 0.5*(bins_out[1:]+bins_out[:-1])
                data['y'] = y
            else:
                data['x'] = np.cumsum(np.ones(len(data['y'])))
            valid_val.set('Valid')
            return data
        except:
            print('Test does not exist')
            valid_val.set('Invalid')
            return None

# the database to load data from
db_name = 'intercepts_scan'
# the name of the intercept scan to look at
test = 'proportion_activity'
save_location = '%s/proportion_time_neurons_active'%test
#save_location = '%s/proportion_neurons_active_over_time'%test
live = LiveFigure(db_name=db_name, save_location=save_location)
app = MasterPage(db_name=db_name, save_location=save_location)#, fig=ani_plot.fig)
ani = animation.FuncAnimation(live.fig, live.plot, interval=1000)
#app.geometry("1280x720")
app.mainloop()
