import abr_jaco2
from abr_analyze.utils import DrawCells, Draw2dData, Draw3dData, DrawArm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
"""
An example combining the different plotting subclasses and plotting them onto
a gridspec grid
"""

# the number of samples to interpolate our data to, set to None for no
# interpolation
interpolated_samples=100
# list our tests and their relevant save locations
db_name = 'abr_analyze'
test = 'my_test_group/test1/session000'
baseline = 'my_test_group/baseline1/session000'

# instantiate our robot config
robot_config = abr_jaco2.Config(use_cython=True, hand_attached=True)

# Instantiate our arm drawing module
draw_arm = DrawArm(db_name=db_name, robot_config=robot_config,
        interpolated_samples=interpolated_samples)

# Instantiate our 2d drawing module
draw_2d = Draw2dData(db_name=db_name,
        interpolated_samples=interpolated_samples)

# Instantiate our 3d drawing module
draw_3d = Draw3dData(db_name=db_name,
        interpolated_samples=interpolated_samples)

# Instantiate our plot organizational module
cells = DrawCells()

# layout our desired plotting grid
outer_grid = gridspec.GridSpec(4,4)
cell1 = outer_grid[0:2,0]
cell2 = outer_grid[0:2,1]
cell3 = outer_grid[2:4,0:2]
cell4 = outer_grid[0:2, 2:4]
cell5 = outer_grid[2:4, 2:4]

# pass our plotting modules, the parameters to plot, and their save locations,
# and the cell we wish to have them plotted onto to our grid organizing module
cells.add_cell(
        cell=cell1,
        function=draw_arm,
        save_locations=['%s/run%03d'%(test,0)]
        )
# the same cell can be passed in multiple times if you would like to use
# different modules on the same cell (draw_arm and draw_3d here)
cells.add_cell(
        cell=cell1,
        function=draw_3d,
        save_locations=['%s/run%03d'%(baseline,0)],
        parameters=['ee_xyz']
        )
cells.add_cell(
        cell=cell2,
        function=draw_arm,
        save_locations=['%s/run%03d'%(test,49)]
        )
cells.add_cell(
        cell=cell2,
        function=draw_3d,
        save_locations=['%s/run%03d'%(baseline,49)],
        parameters=['ee_xyz']
        )
# if the same parameters are to be plotted from the same database, the save
# locations and parameters can be passed in as lists to be plotted on the same
# ax. Alternatively, they can be passed in individually as the above examples
cells.add_cell(
        cell=cell3,
        function=draw_2d,
        save_locations=[
            '%s/run%03d'%(test,0),
            '%s/run%03d'%(test,49)
            ],
        parameters=['q'],
        n_rows=3,
        n_cols=2
        )
# Note that we can specify how many rows and columns to break the cell further
# down into. This can be useful if the data is multi-dimensional. Doing so will
# plot each dimension onto it's own subplot
cells.add_cell(
        cell=cell4,
        function=draw_2d,
        save_locations=[
            '%s/run%03d'%(test,0),
            '%s/run%03d'%(baseline,0)
            ],
        parameters=['u_base'],
        n_rows=6,
        n_cols=1
        )
cells.add_cell(
        cell=cell5,
        function=draw_2d,
        save_locations=[
            '%s/run%03d'%(test,49),
            '%s/run%03d'%(baseline,49)
            ],
        parameters=['u_base'],
        )

# Once we have passed all of our data to our organizational layer, run generate
# to create the plot
cells.generate()
