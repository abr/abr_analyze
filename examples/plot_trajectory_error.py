from abr_analyze.plotting import TrajectoryError
from abr_analyze.paths import figures_dir
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

db_name = 'dewolf2018neuromorphic'
test_group = 'friction_post_tuning'
test_list = [
             'nengo_cpu_friction_102_0',
             'nengo_cpu_friction_103_0',
             'nengo_cpu_friction_104_0',
             'nengo_loihi_friction_46_0',
             'nengo_loihi_friction_47_0',
             'nengo_loihi_friction_48_0',
            ]
label = [
         ': 10k',
         ': 1k',
         ': 50k',
         ': 50k lr30',
         ': 50k lr35',
         ': 50k lr45',
        ]
sessions = 1
runs = 50
time_derivatives = [0,1,2,3]
derivative_print = ['0th', '1st', '2nd', '3rd']
filter_const=None
interpolated_samples=400
if len(time_derivatives)>3:
    plt_dims = [2,int(np.ceil(len(time_derivatives))/2)]
else:
    plt_dims = [len(time_derivatives), 1]

fig = plt.figure(figsize=(12,8))
for jj, td in enumerate(time_derivatives):
    print('-- Processing %s derivative trajectory error --'%derivative_print[td])
    title = 'trajectory_error_%i'%(td)
    ax = fig.add_subplot(plt_dims[0], plt_dims[1], jj+1)
    traj = TrajectoryError(db_name=db_name,
                        time_derivative=td,
                        filter_const=filter_const,
                        interpolated_samples=interpolated_samples)
    c = ['k','b', 'g', 'r', 'y', 'm', 'o', 'tab:purple']
    for ii, test in enumerate(test_list):
        print('Processing test %i/%i: %s'%(ii+1, len(test_list), test))
        traj.statistical_error(save_location='%s/%s'%(test_group, test),
                sessions=sessions, runs=runs, regen=False)
    print('Plotting...')
    for ii, test in enumerate(test_list):
        traj.plot(ax=ax, save_location='%s/%s'%(test_group, test), label=test+label[ii],
                c=c[ii], loc=0, title=title)
        ax.legend()
loc = '%s/trajectory_error.png'%(figures_dir)
print('Figure saved to %s'%(loc))
plt.savefig(loc)
plt.show()