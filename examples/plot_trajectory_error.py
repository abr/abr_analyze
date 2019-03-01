from abr_analyze.utils import TrajectoryError
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

db_name = 'example_db'
test_group = 'friction_post_tuning'
test_list = ['nengo_loihi_friction_9_0', 'nengo_cpu_friction_25_0',
        'nengo_gpu_friction_9_0', 'pd_no_friction_5_0']
sessions = 5
runs = 50
time_derivative=0
filter_const=None
interpolated_samples=100
traj = TrajectoryError(db_name=db_name,
                       time_derivative=time_derivative,
                       filter_const=filter_const,
                       interpolated_samples=interpolated_samples)

plt.figure()
ax = plt.subplot(111)
for test in test_list:
    traj.statistical_eror(save_location=test, sessions=sessions, runs=runs()
for test in test_list:
    traj.plot(ax=ax, save_location=test)
plt.show()
