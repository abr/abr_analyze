"""
A simple example of plotting a 3d stick arm figure from saved data.

The save location must have the joint angles of the robot arm saved under the
key 'q'
"""
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

from abr_control.arms import jaco2

from abr_analyze.plotting import DrawArm, Draw3dData, MakeGif
from abr_analyze.paths import figures_dir
from download_examples_db import check_exists as examples_db


examples_db()
gif = MakeGif()
fig_cache = gif.prep_fig_cache()
# the number of samples to interpolate our data to, set to None for no
# interpolation
interpolated_samples = 100
# list our tests and their relevant save locations
db_name = "abr_analyze_examples"
test = "test_1"
baseline = "baseline_1"

# instantiate our robot config
robot_config = jaco2.Config()

# Instantiate our arm drawing module
draw_arm = DrawArm(
    db_name=db_name,
    robot_config=robot_config,
    interpolated_samples=interpolated_samples,
)

# instantiate our generic trajectory drawing module
draw_3d = Draw3dData(db_name=db_name, interpolated_samples=interpolated_samples)

plt.figure()
ax = plt.subplot(111, projection="3d")
for ii in range(1, interpolated_samples):
    print("%.2f%% complete" % (ii / interpolated_samples * 100), end="\r")
    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(0, 1)
    ax.set_aspect(1)
    draw_3d.plot(
        ax=ax,
        save_location="%s/session000/run000" % baseline,
        parameters="ee_xyz",
        step=ii,
        c="r",
        label="baseline",
    )

    draw_3d.plot(
        ax=ax,
        save_location="%s/session000/run000" % test,
        parameters="ee_xyz",
        step=ii,
        c="b",
        label="test",
    )

    draw_3d.plot(
        ax=ax,
        save_location="%s/session000/run000" % test,
        parameters="ideal_trajectory",
        step=ii,
        c="g",
        linestyle="--",
        label="ideal",
    )

    draw_arm.plot(ax=ax, save_location="%s/session000/run000" % test, step=ii)

    ax.set_title("My 3D Arm Plot")
    save_loc = "%s/%05d.png" % (fig_cache, ii)
    plt.savefig(save_loc)
    ax.clear()

save_loc = "%s" % (figures_dir)
gif.create(
    fig_loc=fig_cache, save_loc=save_loc, save_name="gif_arm", delay=5, res=[1920, 1080]
)
