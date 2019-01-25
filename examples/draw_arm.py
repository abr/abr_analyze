#TODO: make this plot only a single ax object with parameters to either pass an
# ax object, if not one is created since we only want the one frame, otherwise
# get the grid layout done in a higher level script
import abr_jaco2
from abr_analyze.utils import DrawArmProc, DrawArmVis, MakeGif
from abr_analyze.utils.paths import figures_dir
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import os
"""
"""
# True to make several plots over time to turn into gif
animate = True
if animate:
    # instantiate our gif making module
    gif = MakeGif()
    # get the save location for the figures saved over time, and clear it of
    # any previously saved figures.
    fig_cache = gif.prep_fig_cache()
save_name = 'draw-arm-test'

# set the number of samples to take from the data
interpolated_samples = 100

# list our tests and their relevant save locations
db_name = 'example_db'
test_group = 'friction_post_tuning'
test_list = ['nengo_loihi_friction_9_0', 'nengo_cpu_friction_25_0',
        'nengo_gpu_friction_9_0', 'pd_friction_9_0']

# instantiate our robot config
robot_config = abr_jaco2.Config(use_cython=True, hand_attached=True)

# instantiate our drawArm Processor to load baseline data
drawProc = DrawArmProc(db_name=db_name, robot_config=None)
# get our baseline data
baseline = drawProc.generate(save_location=('%s/pd_no_friction_5_0/session000/run000'
        %(test_group)), interpolated_samples=interpolated_samples, clear_memory=True)

# instantiate our drawArm Processor, this time with a robot_config, since our
# comparative data will be plotting an arm as well
drawProc = DrawArmProc(db_name=db_name, robot_config=robot_config)
# get our data to compare to baseline
data= []
for test in test_list:
    save_location='%s/%s/session000/run000'%(test_group, test)
    #NOTE: if clear_memory is true, the instantiated data handling and processing, and
    # robot config objects will be overwritten as None to save memory. This is
    # useful if sending in data from different databases or with different
    # robot_configs. If all the data is from the same data_base and robot_config,
    # it can be left as True to avoid having to reinstantiate DrawArmProc

    # load our data for each specific test
    data.append(drawProc.generate(save_location=save_location,
            interpolated_samples=interpolated_samples, clear_memory=False))

# define our plot grid layout
plt.figure()
ax = []
ax.append(plt.subplot2grid((2,2),(0,0), colspan=1, rowspan=1, projection='3d'))
ax.append(plt.subplot2grid((2,2),(0,1), colspan=1, rowspan=1, projection='3d'))
ax.append(plt.subplot2grid((2,2),(1,0), colspan=1, rowspan=1, projection='3d'))
ax.append(plt.subplot2grid((2,2),(1,1), colspan=1, rowspan=1, projection='3d'))

# instantiate our data visualizer for the baseline
baselineVis = DrawArmVis(traj_color='tab:purple', linestyle='--')

# instantiate our data visualizer for the tests being compared to baseline
# NOTE: each test can instantiate it's own data visualizer to define different
# plotting colors / parameters. In this example, since each test is on it's own
# subplot, we will keep a consistent colour scheme between tests
drawVis = DrawArmVis(traj_color='b', link_color='y', joint_color='k', arm_color='k')

if animate:
    animate_frames = len(data[0]['ee_xyz'])
    steps = range(0, animate_frames)
else:
    steps = [-1]

for step in steps:
    if animate:
        print('%.2f%% complete'%(step/animate_frames*100), end='\r')
    for ii, axis in enumerate(ax):
        # clear the axis so we don't plot on top of the previous frame
        axis.clear()
        axis.set_title(test_list[ii])
        axis.set_xlim3d(-0.5,0.5)
        axis.set_ylim3d(-0.5,0.5)
        axis.set_zlim3d(0,1)
        axis.set_aspect(1)
        drawVis.plot_arm(ax=axis, joints_xyz=data[ii]['joints_xyz'][step],
                links_xyz=data[ii]['links_xyz'][step], ee_xyz=data[ii]['ee_xyz'][step])
        drawVis.plot_trajectory(ax=axis, data=data[ii]['filter'][:step], c='g')
        baselineVis.plot_trajectory(ax=axis, data=baseline['ee_xyz'][:step])
        drawVis.plot_trajectory(ax=axis, data=data[ii]['ee_xyz'][:step])
    if animate:
        plt.savefig('%s/%05d.png'%(fig_cache, step))
    else:
        plt.savefig('%s/%s.png'%(figures_dir, save_name))

if animate:
    gif.create(fig_loc=fig_cache,
               save_loc='%s/%s/draw_arm'%(figures_dir, db_name),
               save_name=save_name,
               delay=5, res=[1920,1080])
else:
    plt.show()
