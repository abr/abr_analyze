"""
An example combining the different plotting subclasses and plotting them onto
a gridspec grid
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from abr_control.arms import jaco2

from abr_analyze.plotting import DrawCells, Draw2dData, Draw3dData, DrawArm
from download_examples_db import check_exists as examples_db


examples_db()
# the number of samples to interpolate our data to, set to None for no
# interpolation
interpolated_samples=100
# list our tests and their relevant save locations
db_name = 'abr_analyze_examples'
test = 'test_1/session000'
baseline = 'baseline_1/session000'

# instantiate our robot config
robot_config = jaco2.Config()

# Instantiate our arm drawing module
draw_arm = DrawArm(
    db_name=db_name,
    robot_config=robot_config,
    interpolated_samples=interpolated_samples)

# Instantiate our 2d drawing module
draw_2d = Draw2dData(
    db_name=db_name,
    interpolated_samples=interpolated_samples)

# Instantiate our 3d drawing module
draw_3d = Draw3dData(
    db_name=db_name,
    interpolated_samples=interpolated_samples)

# Instantiate our plot organizational module
cells = DrawCells()

# layout our desired plotting grid
outer_grid = gridspec.GridSpec(4, 4)
cell1 = outer_grid[0:2, 0]
cell2 = outer_grid[0:2, 1]
cell3 = outer_grid[2:4, 0:2]
cell4 = outer_grid[0:2, 2:4]
cell5 = outer_grid[2:4, 2:4]

# pass our plotting modules, the parameters to plot, and their save locations,
# and the cell we wish to have them plotted onto to our grid organizing module
cells.add_cell(
    cell=cell1,
    function=draw_arm,
    save_location='%s/run%03d'%(test, 0),
    animate=True
    )

# the same cell can be passed in multiple times if you would like to use
# different modules on the same cell (draw_arm and draw_3d here)
cells.add_cell(
    cell=cell1,
    function=draw_3d,
    save_location='%s/run%03d'%(test, 0),
    parameters=['ee_xyz'],
    animate=True
    )

cells.add_cell(
    cell=cell1,
    function=draw_3d,
    save_location='%s/run%03d'%(baseline, 0),
    parameters=['ee_xyz'],
    animate=True
    )
cells.add_cell(
    cell=cell2,
    function=draw_arm,
    save_location='%s/run%03d'%(test, 9)
    )
cells.add_cell(
    cell=cell2,
    function=draw_3d,
    save_location='%s/run%03d'%(baseline, 9),
    parameters=['ee_xyz']
    )

# Each save location gets its own call, however the parameters to plot can be
# passed in as a list if multiple are to be plotted in the same location
cells.add_cell(
    cell=cell3,
    function=draw_2d,
    save_location='%s/run%03d'%(test, 0),
    parameters=['q', 'time'],
    subplot=[3,2]
    )
cells.add_cell(
    cell=cell3,
    function=draw_2d,
    save_location='%s/run%03d'%(test, 9),
    parameters=['time', 'q'],
    subplot=[3,2]
    )

# Note that we can specify how many rows and columns to break the cell further
# down into. This can be useful if the data is multi-dimensional. Doing so will
# plot each dimension onto it's own subplot
cells.add_cell(
    cell=cell4,
    function=draw_2d,
    save_location='%s/run%03d'%(test, 0),
    parameters=['q'],
    subplot=[6,1]
    )

# however, if you try to change the number or rows and columns on a cell that
# has already been passed in, only the initial set values will be used
cells.add_cell(
    cell=cell4,
    function=draw_2d,
    save_location='%s/run%03d'%(baseline, 0),
    parameters=['q'],
    subplot=[3,2]
    )

# if no rows or columns are specified, then everything will be plotted on 1 ax
cells.add_cell(
    cell=cell5,
    function=draw_2d,
    save_location='%s/run%03d'%(test, 9),
    parameters=['ideal_trajectory'],
    )
cells.add_cell(
    cell=cell5,
    function=draw_2d,
    save_location='%s/run%03d'%(baseline, 9),
    parameters=['ideal_trajectory'],
    )

# Once we have passed all of our data to our organizational layer,
# run generate to create the plot
cells.generate()
