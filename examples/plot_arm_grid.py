"""
Accepts a save location for arm tests and a list of runs to plot, plotting them
on different cells
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

from abr_analyze.utils.paths import figures_dir
from abr_analyze.utils import DrawCells, Draw3dData, DrawArm
import abr_jaco2

interpolated_samples = 100
db_name='dewolf2018neuromorphic'
group = 'friction_post_tuning'
animate = True
# assert animate is True and interpolated_samples is None, (
#         "You must interpolate data for animation")
tests = [
        'nengo_loihi_friction_45_0',
        'nengo_loihi_friction_44_0'
        ]
runs_list = [range(0,17),
             range(17,35),
             range(35,50)]
session = 0
rows = 3
cols = 6
save_names = ['arm_grid_0-16',
              'arm_grid_17-34',
              'arm_grid_35-49']
for cnt, runs in enumerate(runs_list):
    while rows*cols < len(runs):
        rows += 1

    # instantiate our robot config
    robot_config = abr_jaco2.Config(use_cython=True, hand_attached=True)

    # Instantiate our arm drawing module
    draw_arm = DrawArm(
        db_name=db_name,
        robot_config=robot_config,
        interpolated_samples=interpolated_samples,
        show_trajectory=False)

    # Instantiate our 3d drawing module
    draw_3d = Draw3dData(
        db_name=db_name,
        interpolated_samples=interpolated_samples)

    # Instantiate our plot organizational module
    draw_cells = DrawCells()

    # layout our desired plotting grid
    outer_grid = gridspec.GridSpec(rows, cols)
    cells = []
    for run in range(0, len(runs)):
        cells.append(outer_grid[int(run/cols),run%cols])

    for ii in range(0, len(runs)):
        draw_cells.add_cell(
                cell=cells[ii],
                function=draw_3d,
                save_location=('%s/%s/session%03d/run%03d'
                    %(group, tests[1], session, runs[ii])),
                parameters=['ee_xyz'],
                c='tab:red',
                label=tests[1],
                title=runs[ii],
                animate=animate
                )
        draw_cells.add_cell(
                cell=cells[ii],
                function=draw_arm,
                save_location=('%s/%s/session%03d/run%03d'
                    %(group, tests[0], session, runs[ii])),
                c='tab:blue',
                label=None,
                animate=animate
                )
        draw_cells.add_cell(
                cell=cells[ii],
                function=draw_3d,
                save_location=('%s/%s/session%03d/run%03d'
                    %(group, tests[0], session, runs[ii])),
                parameters=['filter'],
                c='tab:green',
                linestyle='--',
                label='ideal',
                animate=animate
                )
        draw_cells.add_cell(
                cell=cells[ii],
                function=draw_3d,
                save_location=('%s/%s/session%03d/run%03d'
                    %(group, tests[0], session, runs[ii])),
                parameters=['ee_xyz'],
                c='tab:blue',
                linestyle='-',
                label=tests[0],
                animate=animate
                )

    draw_cells.generate(save_name=save_names[cnt])
