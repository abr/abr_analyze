"""
A simple example of plotting a 3d stick arm figure from saved data.

The save location must have the joint angles of the robot arm saved under the
key 'q'
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

from abr_control.arms import jaco2

from abr_analyze.plotting import DrawArm, Draw3dData
from abr_analyze.paths import figures_dir
from download_examples_db import check_exists as examples_db


examples_db()
# the number of samples to interpolate our data to, set to None for no
# interpolation
interpolated_samples=100
# list our tests and their relevant save locations
db_name = 'abr_analyze_examples'
test = 'test_1'
baseline = 'baseline_1'

# instantiate our robot config
robot_config = jaco2.Config()

# Instantiate our arm drawing module
draw_arm = DrawArm(db_name=db_name, robot_config=robot_config,
        interpolated_samples=interpolated_samples)

# instantiate our generic trajectory drawing module
draw_3d = Draw3dData(db_name=db_name,
        interpolated_samples=interpolated_samples)

plt.figure()
ax = plt.subplot(111, projection='3d')

draw_3d.plot(
        ax=ax,
        save_location='%s/session000/run000'%baseline,
        parameters='ee_xyz',
        label='baseline')

draw_arm.plot(
        ax=ax,
        save_location='%s/session000/run000'%test)

draw_3d.plot(
        ax=ax,
        save_location='%s/session000/run000'%test,
        parameters='ideal_trajectory',
        label='ideal',
        c='g',
        linestyle='--')

draw_3d.plot(
        ax=ax,
        save_location='%s/session000/run000'%test,
        parameters='ee_xyz',
        label='test',
        c='b')

plt.title('3D Arm Plot')
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(0.0, 1.2)

save_loc='%s/3d_arm_plot.png'%(figures_dir)
plt.savefig(save_loc)
plt.show()
print('Saved to %s'%save_loc)
