from abr_analyze.plotting import TrajectoryError
from abr_analyze.paths import figures_dir
from download_examples_db import check_exists as examples_db
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

examples_db()
db_name = 'abr_analyze_examples'
test_list = [
             'test_1',
             'baseline_1',
            ]
label = [
         '',
         '',
        ]
sessions = 5
runs = 10
time_derivatives = [0,1,2,3]
derivative_print = ['0th', '1st', '2nd', '3rd']
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
                        interpolated_samples=interpolated_samples)
    for ii, test in enumerate(test_list):
        print('Processing test %i/%i: %s'%(ii+1, len(test_list), test))
        traj.statistical_error(save_location='%s'%(test),
                sessions=sessions, runs=runs, regen=True)
    print('Plotting...')
    c = ['b', 'r']
    for ii, test in enumerate(test_list):
        traj.plot(ax=ax, save_location='%s'%(test), label=test+label[ii],
                c=c[ii], loc=0, title=title)
        ax.legend()
loc = '%s/examples/trajectory_error.png'%(figures_dir)
print('Figure saved to %s'%(loc))
plt.savefig(loc)
plt.show()
