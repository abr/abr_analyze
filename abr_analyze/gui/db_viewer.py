# Add link to sentdex youtube
# https://www.youtube.com/watch?v=A0gaXfM1UN0&index=2&list=PLQVvvaa0QuDclKx-QpC9wntnURXVJqLyk
# TODO Tutorial 19 adds help button option to walk through gui
"""
sudo apt-get build-dep python-imaging
sudo apt-get install libjpeg8 libjpeg62-dev libfreetype6 libfreetype6-dev
sudo pip install Pillow
"""
import sys
import tkinter as tk
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")
from os import listdir
from os.path import isfile, join

import matplotlib.animation as animation
import numpy as np
import seaborn
from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
)  # , NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk
from terminaltables import AsciiTable

from abr_analyze import DataHandler
from abr_analyze.paths import database_dir

np.set_printoptions(threshold=12)
folder = None
db = None
# if len(sys.argv) >= 2:
for val in sys.argv:
    if "--folder" in val:
        folder = val.split("==")[-1]
        database_dir = folder
    if "--name" in val:
        db = val.split("==")[-1]

if db is None:
    onlyfiles = [f for f in listdir(database_dir) if isfile(join(database_dir, f))]
    print(
        f"No database passed in, the following are available in the repo database direction: {database_dir}"
    )
    for ii, fname in enumerate(onlyfiles):
        print(f"{ii}) {fname}")
    index = input("Which databse would you like to view?")
    db = onlyfiles[int(index)].split(".")[0]

# else:
#     db = sys.argv[1]
# if len(sys.argv)>2:
#     folder = sys.argv[2]

dat = DataHandler(db_name=db, database_dir=folder)


class bcolors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


# set some constants
LARGE_FONT = ("Verdana", 20)
MED_FONT = ("Verdana", 10)
SMALL_FONT = ("Verdana", 8)
style.use("ggplot")
global text_color
# charcoal grey
text_color = "#36454f"
global background_color
background_color = "white"
global button_color
# abr blue
button_color = "#375580"
global button_text_color
button_text_color = "#d3d3d3"  #'white'
secondary_text = "black"
secondary_bg = "#ffff7f"

f = Figure(figsize=(10, 12), dpi=100)
a = f.add_subplot(111)

# global variable for searching in db
loc = ["/"]
# list of tests to display data from in plot
disp_loc = []
tests_plotted = []
# sets whether to display data passed the group 'session'
browse_datasets = True
# list of possible plotting variables
# NOTE not being used atm, needs to be updated
plotting_variables = []
global plotting_colors
plotting_colors = []
# boolean that triggers when a test is added or removed from the plotting list
update_plot = False
# list of selected variables to plot
var_to_plot = "avg error"
# last_plotted = var_to_plot
# variable for toggling whether to save current figure
save_figure = False
# NOTE not being used atm, needs to be updated
orders_of_error = []
# orders_of_error = ['position', 'velocity', 'acceleration', 'jerk']
global order_to_plot
order_to_plot = "position"
global param_to_compare
param_to_compare = None
# create radio buttons at bottom of search list for all params plotted
# selecting the radio button will print a table comparing params of tests
global modules
modules = ["params"]


def live_plot(i):
    """
    The function that plots the selected tests and data
    """
    global save_figure
    global plotting_colors
    global update_plot
    global disp_loc
    global tests_plotted

    if update_plot:
        # used for setting the legend for multi line data
        multi_line = False
        line_styles = ["-", "--", "-.", ":"]

        # cycle through selected tests to plot
        legend_names = []
        # print('STARTING DISP: ', disp_loc)
        tests_to_remove = []
        plotting_colors = ["r", "g", "b", "o", "y", "m", "k", "tab:purple", "tab:grey"]
        # print('disp_loc: ', disp_loc)
        max_dims = 1
        max_len = 1

        tests = {}
        for count, test in enumerate(disp_loc):
            splt_loc = test.split("/")
            splt_loc = [string for string in splt_loc if string != ""]
            # TODO will need a check here if there is no folder structure and splt_loc is just the param
            var_to_plot = splt_loc[-1]
            location = "/".join(splt_loc[:-1])

            if location not in tests:
                legend_names.append(location)
                tests[location] = [var_to_plot]
                if location not in tests_plotted:
                    tests_plotted.append(location)
            else:
                tests[location].append(var_to_plot)

            d = dat.load(parameters=[var_to_plot], save_location="%s" % (location))
            d = np.array(d[var_to_plot])

            if len(d.shape) == 1:
                d = np.expand_dims(d, axis=1)

            max_dims = max(max_dims, d.shape[1])
            max_len = max(max_len, d.shape[0])

        f.clear()
        axs = []
        for ii in range(max_dims):
            rows = min(6, max_dims)
            axs.append(f.add_subplot(int(rows), int(np.ceil(max_dims / rows)), ii + 1))

        for a in axs:
            a.clear()

        y_maxs = np.zeros(max_dims)
        # something large so we overwrite it with the actual data
        y_mins = np.ones(max_dims) * 1e6
        x_maxs = np.ones(max_dims) * max_len

        for count, location in enumerate(tests):
            # TODO will need a check here if there is no folder structure and splt_loc is just the param
            for var_to_plot in tests[location]:
                d = dat.load(parameters=[var_to_plot], save_location="%s" % (location))
                d = np.array(d[var_to_plot])

                if len(d.shape) == 1:
                    d = np.expand_dims(d, axis=1)

                for ii, dim in enumerate(d.T):
                    if max(dim) > y_maxs[ii]:
                        y_maxs[ii] = max(dim)
                    if min(dim) < y_mins[ii]:
                        y_mins[ii] = min(dim)
                    if len(dim) > x_maxs[ii]:
                        x_maxs[ii] = len(dim)

                    a = axs[ii]
                    a.set_xlim(0, x_maxs[ii])
                    a.set_ylim(y_mins[ii], y_maxs[ii])
                    a.plot(dim, label="%s_%i" % (var_to_plot, ii))

        for a in axs:
            a.legend(loc=2)  # bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
        f.tight_layout()

        for rm_test in tests_to_remove:
            disp_loc.remove(rm_test)

        if save_figure:
            # f.savefig('%s.pdf'%var_to_plot)
            f.savefig("db_viewer_fig.png")
            save_figure = False

        update_plot = False


def save_figure_toggle(self):
    """
    a toggle that is changed based on a button click. it is momentarily
    toggled to save the current figure
    """
    global save_figure
    global update_plot
    save_figure = True
    update_plot = True
    print("Figure Saved")


def clear_plot(self):
    """
    a toggle that is changed based on a button click and is used to clear the
    plot
    """
    global disp_loc
    global update_plot
    global tests_plotted
    disp_loc = []
    tests_plotte = []
    update_plot = True


def popupmsg(msg):
    """
    generic function to pass in a string to appear in a popup message
    """
    popup = tk.Tk()
    popup.wm_title("!")
    label = tk.Label(popup, text=msg, font=MED_FONT)
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="OK", command=popup.destroy)
    B1.pack()
    popup.mainloop()


def go_back_loc_level(self):
    """
    function used to go back a search level, can be thought of as going back
    a directory
    """
    global loc
    loc = loc[:-1]
    # self.entry.delete(0, 'end')
    self.update_list()


def go_to_root_level(self):
    """
    Function used to reset database location to root location
    """
    global loc
    loc = ["/"]
    self.update_list()


def toggle_browse_datasets(self):
    """
    Toggles the browse_datasets variable.
    If browse datasets is set to False
    we will stop going down the chain in the database once we reach a group
    that contains a 'session' group. At this point we want to save the
    selected test and plot it.
    If browse datasets is set to True then we will go further down the chain,
    passed the session group level and allow the user to view the save
    structure.
    """
    global browse_datasets
    browse_datasets = not browse_datasets


class Page(tk.Tk):
    def __init__(self, *args, **kwargs):

        # instantiate our container
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "abr_control data")

        container = tk.Frame(self)
        # container.configure(background='white')
        container.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # menu bar at top
        menubar = tk.Menu(container)
        # define main file menu
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(
            label="Save Figure", command=lambda: save_figure_toggle(self)
        )
        # command=lambda:popupmsg(msg="Not Supported Yet"))
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=quit)
        # place file menu
        menubar.add_cascade(label="File", menu=filemenu)

        # Example of another menubar entry that is not currently used
        # # define parameter to plot menu
        # self.param_menu = tk.Menu(menubar, tearoff=1)
        # self.param_menu.add_command(label="None",
        #         command=lambda:popupmsg(
        #             msg=("There are no parameters to select from from the"
        #             + " current database group")))
        # self.param_menu.add_command(label="Error",
        #         command=lambda:changeParam("error"))
        # # place parameter menu bar
        # menubar.add_cascade(label="Plotting Parameters", menu=self.param_menu)

        tk.Tk.config(self, menu=menubar)

        self.frames = {}

        # place our other pages into the master page
        for F in (StartPage, SearchPage):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)

    def show_frame(self, cont):
        """
        Function for pulling the selected frame to the front
        """
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):
    """
    The starting page when the gui is loaded
    """

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.configure(background=background_color)

        page_title = tk.Label(self, text="Home", font=LARGE_FONT)
        page_title.grid(row=0, column=1, padx=10)
        page_title.configure(background=background_color, foreground=text_color)

        main_text = tk.Label(
            self,
            text=(
                "Welcome to the abr_control plotting GUI."
                + "\n\nBrowse through your recorded tests in the 'Search' page by"
                + '\nselecting the blue "Search" button below.'
                + "\nAs you select datasets from the list they will appear in the plot."
                + "\nSelect them again to remove them, or click the 'Clear Plot'"
                + "\nbutton to clear the plot entirely."
                + "\nTo save your figure, from the upper menu select File>Save Figure"
                + "\n\nWhile collecting data, save your test parameters in a params folder,"
                + "\nlocated where your data is saved. You can further break this down into"
                + '\nsubfolders. Once you have plotted some tests you can select the "params"'
                + "\nbutton below the search bar to get a table printout of the parameters"
                + "\nused in the plotted tests. If there are subfolders they will be added"
                + '\nto the radio button list once the "params" button has been selected.'
                + "\nThe subfolders can be recursive, however they will only be added to the"
                + "\nthe radio button list once the parent folder has been selected/loaded."
                + "\nThe table will plot all the parameters in the selected params folder,"
                + "\nand will highlight differences between the tests plotted"
            ),
            font=MED_FONT,
        )
        main_text.grid(row=1, column=1)
        main_text.configure(background=background_color, foreground=text_color)

        # Add a button to take bring the search page to the front
        search_button = tk.Button(
            self,
            text="Search",
            command=lambda: controller.show_frame(SearchPage),
            bg=button_color,
            fg=button_text_color,
        )
        search_button.grid(row=2, column=1)


class SearchPage(tk.Frame):
    def __init__(self, parent, controller):
        # instantiate our frame
        tk.Frame.__init__(self, parent)
        self.configure(background=background_color)

        # create a left and right frame to simplify organization of grid
        frame_top = tk.Frame(self, parent)
        frame_top.grid(row=0, column=0, padx=10)
        frame_top.configure(background=background_color)

        # the bottom frame will be used for the parameter comparison table
        self.frame_bottom = tk.Frame(self, parent)
        self.frame_bottom.grid(row=1, column=0, padx=10)
        self.frame_bottom.configure(background=background_color)

        frame_left = tk.Frame(self, frame_top)
        frame_left.grid(row=0, column=0, padx=10)
        frame_left.configure(background=background_color)

        frame_right = tk.Frame(self, frame_top)
        frame_right.grid(row=0, column=2, padx=10)
        frame_right.configure(background=background_color)

        frame_right_top = tk.Frame(frame_right)
        frame_right_top.grid(row=0, column=0, padx=10, pady=10)
        frame_right_top.configure(background=background_color)

        frame_right_bottom = tk.Frame(frame_right)
        frame_right_bottom.grid(row=1, column=0, padx=10, pady=10)
        frame_right_bottom.configure(background=background_color)

        page_title = tk.Label(frame_left, text="Search", font=LARGE_FONT)
        page_title.grid(row=0, column=1)
        page_title.configure(background=background_color, foreground=text_color)

        self.current_location_display = tk.StringVar()
        self.current_location_display.set("".join(loc))

        # text printout to show our current search location
        current_location_label = tk.Label(
            frame_left, textvariable=self.current_location_display, font=MED_FONT
        )
        current_location_label.grid(row=2, column=0, columnspan=3)
        current_location_label.configure(
            background=background_color, foreground=text_color
        )

        # create our buttons
        home_button = tk.Button(
            frame_left, text="Home", command=lambda: go_to_root_level(self)
        )
        # command=lambda: controller.show_frame(StartPage))
        home_button.grid(row=1, column=1)  # , sticky='nsew')
        home_button.configure(background=button_color, foreground=button_text_color)

        back_button = tk.Button(
            frame_left, text="Back", command=lambda: go_back_loc_level(self)
        )
        back_button.grid(row=1, column=0)  # , sticky='nsew')
        back_button.configure(foreground=button_text_color, background=button_color)

        clear_plot_button = tk.Button(
            frame_left, text="Clear Plot", command=lambda: clear_plot(self)
        )
        clear_plot_button.grid(row=1, column=2)  # , sticky='nsew')
        clear_plot_button.configure(
            background=button_color, foreground=button_text_color
        )

        browse_datasets_button = tk.Button(
            frame_left,
            text="Browse Datasets",
            command=lambda: toggle_browse_datasets(self),
        )
        browse_datasets_button.grid(row=5, column=1)  # , sticky='nsew')
        browse_datasets_button.configure(
            foreground=button_text_color, background=button_color
        )

        # create our search bar and list box
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self.update_list)
        self.entry = tk.Entry(frame_left, textvariable=self.search_var, width=13)
        self.lbox = tk.Listbox(frame_left, width=45, height=15, selectmode="MULTIPLE")
        self.lbox.bind("<<ListboxSelect>>", self.get_selection)

        self.entry.grid(
            row=3, column=0, columnspan=3, sticky="ew"
        )  # , sticky='nsew', columnspan=3)
        self.lbox.grid(row=4, column=0, columnspan=3)  # , sticky='nsew', columnspan=3)
        self.entry.configure(background=background_color, foreground=text_color)
        self.lbox.configure(background=background_color, foreground=text_color)

        # Function for updating the list/doing the search.
        # It needs to be called here to populate the listbox.
        self.update_list()
        # values = [self.lbox.get(idx) for idx in self.lbox.curselection()]

        # Plotting Window
        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=1)  # , ipadx=100, sticky='w')
        canvas.get_tk_widget().configure(background=background_color)

        # show the matplotlib toolbar
        # toolbar = NavigationToolbar2TkAgg(canvas, self)
        # toolbar.update()
        # canvas._tkcanvas.grid(row=0, column=5, columnspan=10)

        # initialize radio buttons
        self.var_to_plot = tk.StringVar()
        self.var_to_plot.set(var_to_plot)
        for ii, var in enumerate(plotting_variables):
            var_to_plot_radio = tk.Radiobutton(
                frame_right_top,
                text=var,
                variable=self.var_to_plot,
                value=var,
                command=lambda: self.update_var_to_plot(self),
            )
            var_to_plot_radio.grid(
                row=ii, column=0, ipadx=20, sticky="ew"
            )  # , sticky='nsew')
            if var == "mean & ci":
                var_to_plot_radio.configure(
                    background=secondary_bg, foreground=secondary_text
                )
            else:
                var_to_plot_radio.configure(
                    background=button_color, foreground=button_text_color
                )

        # variable to track which parameters to compare
        self.param_to_compare_selection = tk.StringVar()
        self.param_to_compare_selection.set(None)

        # variable to track what order of error to plot
        self.order_to_plot = tk.StringVar()
        self.order_to_plot.set(order_to_plot)
        self.order_radio_button = []
        for ii, order in enumerate(orders_of_error):
            self.order_radio_button.append(
                tk.Radiobutton(
                    frame_right_bottom,
                    text=order,
                    variable=self.order_to_plot,
                    value=order,
                    command=lambda: self.update_order_to_plot(self),
                )
            )
            self.order_radio_button[ii].grid(row=ii, column=0, ipadx=20, sticky="ew")
            self.order_radio_button[ii].configure(
                background="red", foreground=secondary_text
            )
            # order_radio_button.visible = False

        # create the list of parameter module radio buttons
        self.show_params()

    def update_var_to_plot(self, *args):
        """updates the global variable of what data to plot"""
        global var_to_plot
        var_to_plot = self.var_to_plot.get()
        if var_to_plot == "mean & ci":
            col = secondary_bg
        else:
            col = "red"
        for ii, order in enumerate(orders_of_error):
            self.order_radio_button[ii].configure(background=col)

    def update_order_to_plot(self, *args):
        global order_to_plot
        global update_plot
        order_to_plot = self.order_to_plot.get()
        update_plot = True
        self.update_list()

    def get_selection(self, *args):
        """
        get the selection from the listbox and update the list and search
        location accordingly based on button selection and location
        """
        global loc
        global disp_loc
        global plotting_colors
        global update_plot
        # global data_processed

        # delete the current search
        # self.entry.delete(0, 'end')
        # get cursor selection and update db search location
        index = int(self.lbox.curselection()[0])
        value = self.lbox.get(index)
        # print('You selected item %d: "%s"' % (index, value))
        # append the most recent selection to our search location
        loc.append("%s/" % value)

        # check if we're pointing at a dataset, if so go back one level
        # print('FIRST ATTEMPT TO LOAD: ', ''.join(loc))
        keys = dat.get_keys("".join(loc))
        # if keys are None, then we are pointing at a dataset
        if len(keys) == 1 and keys[0] is None:
            # if the user selects the browse_datasets button, then they want to
            # view the save structure passed the session group level
            if browse_datasets:
                # print('deep dive: loading ', (loc[-1])[:-1])
                # print('looking in: ', ''.join(loc[:-1]))
                # print('PARAMS: ', loc[-1][:-1])
                # print('LOC: ', ''.join(loc[:-1]))
                # print("CURRENT DISPLAY LIST: ", disp_loc)
                var_to_plot = loc[-1][:-1]
                loaded_data = dat.load(
                    parameters=[(loc[-1])[:-1]], save_location="".join(loc[:-1])
                )
                # print the selected dataset to the terminal
                for browse_key in loaded_data:
                    print("Folder: ", "".join(loc[:-1]))
                    print("Size: ", np.array(loaded_data[browse_key]).shape)
                    print("\n", loaded_data)

                test_name = "".join(loc)

                if test_name in disp_loc:
                    index = disp_loc.index(test_name)
                    disp_loc.remove(test_name)
                    # remove the entry of plotting colors that corresponds to
                    # the test being removed
                    del plotting_colors[index]
                    # del data_processed[index]
                else:
                    disp_loc.append("".join(loc))
                    # data_processed.append(processed)

                # print("CURRENT DISPLAY LIST: ", disp_loc)
                update_plot = True
                go_back_loc_level(self)

        # if the selection takes us to the next level of groups then erase the
        # search bar
        # else:
        self.update_list()

    def data_processed(self, loc, *args):
        search = dat.get_keys(loc)
        if search is not None and any("proc_data" in group for group in search):
            if any(
                order_to_plot in group for group in dat.get_keys("%s/proc_data" % loc)
            ):
                # print('%s found in %s'%(order_to_plot, loc))
                return True
            else:
                return False
        else:
            return False

    def update_list(self, *args):
        """
        Function that updates the listbox based on the current search location
        """
        global loc
        self.current_location_display.set("".join(loc))
        search_term = self.search_var.get()

        # pull keys from the database
        lbox_list = dat.get_keys("".join(loc))

        self.lbox.delete(0, tk.END)

        for ii, item in enumerate(lbox_list):
            if search_term.lower() in item.lower():
                self.lbox.insert(tk.END, item)

            if not browse_datasets:
                search = "%s%s" % ("".join(loc), item)
                if self.data_processed(loc=search):
                    self.lbox.itemconfig(ii, bg=secondary_bg, foreground=secondary_text)

    def show_params(self, *args):
        global modules
        module_button = []
        for ii, module in enumerate(modules):
            module_button.append(
                tk.Radiobutton(
                    self.frame_bottom,
                    text=module,
                    variable=self.param_to_compare_selection,
                    value=module,
                    command=lambda: self.update_param_to_compare(self),
                )
            )
        for ii, button in enumerate(module_button):
            button.grid(row=int(np.floor(ii / 3)), column=ii % 3, sticky="ew")
            button.configure(foreground=button_text_color, background=button_color)

    def update_param_to_compare(self, *args):
        global param_to_compare
        global tests_plotted
        global modules
        param_to_compare = self.param_to_compare_selection.get()
        # get selected module parameters and print them out
        update_param_buttons = False
        if param_to_compare is not None:
            test_data = []
            test_params = []

            for ii, test in enumerate(tests_plotted):
                module_keys = dat.get_keys("%s/%s" % (test, param_to_compare))
                dataset_keys = []
                for jj, key in enumerate(module_keys):
                    # only add to list if is a folder, not a dataset (folders only)
                    if not dat.is_dataset("%s/%s/%s" % (test, param_to_compare, key)):
                        # if we haven't created a button for it, add to list
                        if "%s/%s/" % (param_to_compare, key) not in modules:
                            modules.append("%s/%s/" % (param_to_compare, key))
                            update_param_buttons = True
                    else:
                        # if is a dataset, add it to our keys to load data for
                        dataset_keys.append(key)
                        # if not in our master key list for the current table, add it
                        if key not in test_params:
                            test_params.append(key)
                # load the parameter dictionary and save it
                module_data = dat.load(
                    parameters=dataset_keys,
                    save_location="%s/%s" % (test, param_to_compare),
                )
                test_data.append(module_data)

            if update_param_buttons:
                self.show_params()

            table_data = []
            table_header = ["Test Name"] + test_params

            # set the table header colour
            for mm, header in enumerate(table_header):
                table_header[mm] = bcolors.HEADER + header + bcolors.ENDC

            # extract the data from the dictionaries
            formatted_data = []
            for ii, test_dict in enumerate(test_data):
                temp_data = []
                for key in test_params:
                    try:
                        data = test_dict[key]
                        if ii > 0 and np.all(test_data[0][key] != data):
                            data = "%s%s%s" % (bcolors.YELLOW, data, bcolors.ENDC)
                    except KeyError:
                        data = "%sNone%s" % (bcolors.RED, bcolors.ENDC)
                    temp_data.append(data)
                formatted_data.append([tests_plotted[ii]] + temp_data)

            table_data.append(table_header)
            for row in formatted_data:
                table_data.append(row)
            table = AsciiTable(table_data)
            print(table.table)


app = Page()
# app.geometry("1280x720")
ani = animation.FuncAnimation(f, live_plot, interval=1000)
app.mainloop()
