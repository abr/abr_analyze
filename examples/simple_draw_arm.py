#TODO: make this plot only a single ax object with parameters to either pass an
# ax object, if not one is created since we only want the one frame, otherwise
# get the grid layout done in a higher level script
import abr_jaco2
from abr_analyze.utils import DrawArm, MakeGif, DataVisualizer, Draw3dData
from abr_analyze.utils.paths import figures_dir
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
"""
"""
interpolated_samples=100
animate=False
if animate:
    gif = MakeGif()
    fig_cache = gif.prep_fig_cache()

# list our tests and their relevant save locations
db_name = 'example_db'
test = 'friction_post_tuning/nengo_loihi_friction_9_0'
baseline = 'friction_post_tuning/pd_friction_9_0'

# instantiate our robot config
robot_config = abr_jaco2.Config(use_cython=True, hand_attached=True)

# Instantiate our arm drawing module
draw_arm = DrawArm(db_name=db_name, robot_config=robot_config,
        interpolated_samples=interpolated_samples)

# instantiate our generic trajectory drawing module
draw_3d = Draw3dData(db_name=db_name,
        interpolated_samples=interpolated_samples)

# Instantiate our general dataVisualizer
vis = DataVisualizer()

if animate:
    steps = range(1,interpolated_samples)
else:
    steps = [-1]

plt.figure()
ax = plt.subplot(111, projection='3d')

for step in steps:
    ax.clear()
    if animate:
        print('%.2f%% complete'%(step/interpolated_samples*100), end='\r')
    draw_3d.plot(
            ax=ax,
            save_location=['%s/session000/run000'%baseline,
                           '%s/session000/run044'%baseline],
            param='ee_xyz',
            step=step)

    draw_arm.plot(
            ax=ax,
            save_location='%s/session000/run000'%test,
            step=step)

    if animate:
        save_loc = '%s/%05d.png'%(fig_cache, step)
    else:
        save_loc='%s/%s/draw_arm/draw_arm_test.png'%(figures_dir, db_name)
    plt.savefig(save_loc)

if animate:
    save_loc='%s/%s/draw_arm'%(figures_dir, db_name)
    gif.create(fig_loc=fig_cache,
               save_loc=save_loc,
               save_name='draw_arm_test',
               delay=5, res=[1920,1080])
else:
    plt.show()
print('Saved to %s'%save_loc)
