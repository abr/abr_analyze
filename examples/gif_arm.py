import abr_jaco2
from abr_analyze.plotting import DrawArm, Draw3dData, MakeGif
from abr_analyze.paths import figures_dir
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
"""
A simple example of plotting a 3d stick arm figure from saved data.

The save location must have the joint angles of the robot arm saved under the
key 'q'
"""
gif = MakeGif()
fig_cache = gif.prep_fig_cache()
# the number of samples to interpolate our data to, set to None for no
# interpolation
interpolated_samples=100
# list our tests and their relevant save locations
db_name = 'abr_analyze'
test = 'my_test_group/test1'
baseline = 'my_test_group/baseline1'

# instantiate our robot config
robot_config = abr_jaco2.Config(use_cython=True, hand_attached=True)

# Instantiate our arm drawing module
draw_arm = DrawArm(db_name=db_name, robot_config=robot_config,
        interpolated_samples=interpolated_samples)

# instantiate our generic trajectory drawing module
draw_3d = Draw3dData(db_name=db_name,
        interpolated_samples=interpolated_samples)

plt.figure()
ax = plt.subplot(111, projection='3d')
for ii in range(1, interpolated_samples):
    print('%.2f%% complete'%(ii/interpolated_samples*100), end='\r')
    ax.set_xlim3d(-0.5,0.5)
    ax.set_ylim3d(-0.5,0.5)
    ax.set_zlim3d(0,1)
    ax.set_aspect(1)
    draw_3d.plot(
            ax=ax,
            save_location='%s/session000/run000'%baseline,
            parameters='ee_xyz',
            step=ii)

    draw_arm.plot(
            ax=ax,
            save_location='%s/session000/run000'%test,
            step=ii)

    ax.set_title('My 3D Arm Plot')
    save_loc = '%s/%05d.png'%(fig_cache, ii)
    plt.savefig(save_loc)
    ax.clear()

save_loc='%s/examples'%(figures_dir)
gif.create(fig_loc=fig_cache,
            save_loc=save_loc,
            save_name='gif_arm',
            delay=5, res=[1920,1080])
print('Gif saved to %s'%save_loc)
