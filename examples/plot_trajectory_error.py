from abr_analyze.plotting import TrajectoryError
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
          ]
sessions = 1
runs = 50
time_derivative=0
filter_const=None
interpolated_samples=400
traj = TrajectoryError(db_name=db_name,
                       time_derivative=time_derivative,
                       filter_const=filter_const,
                       interpolated_samples=interpolated_samples)

fig = plt.figure()
ax = fig.add_subplot(111)
c = ['k','b', 'g', 'r', 'y']
for ii, test in enumerate(test_list):
    print('Processing test %i/%i: %s'%(ii+1, len(test_list), test))
    traj.statistical_error(save_location='%s/%s'%(test_group, test),
            sessions=sessions, runs=runs)
for ii, test in enumerate(test_list):
    print('Plotting...')
    traj.plot(ax=ax, save_location='%s/%s'%(test_group, test), label=test,
            c=c[ii], loc=0)
    ax.legend()
# ax = [plt.subplot(211),
#         plt.subplot(212)]
# tenk_tests = [
#               test_list[0],
#               test_list[1],
#               test_list[2],
#               test_list[4],
#               ]
#
# fiftyk_tests = [
#               test_list[0],
#               test_list[1],
#               test_list[3],
#               test_list[5],
#               ]
#
# for ii, test in enumerate(tenk_tests):
#     print('Plotting...')
#     traj.plot(ax=ax[0], save_location='%s/%s'%(test_group, test), label=test,
#             c=c[ii], loc=0, title='10k')
# ax[0].legend(loc=2)
# for ii, test in enumerate(fiftyk_tests):
#     print('Plotting...')
#     traj.plot(ax=ax[1], save_location='%s/%s'%(test_group, test), label=test,
#             c=c[ii], loc=0, title='50k')
# ax[1].legend(loc=2)
plt.show()
