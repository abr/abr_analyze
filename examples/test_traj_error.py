from abr_analyze.utils import (TrajectoryErrorProc, DataHandler, DataProcessor)
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec

dat = DataHandler(db_name='example_db')

# x = np.ones(628) * 0.01
# y = np.sin(np.cumsum(x))
# plt.figure()
# yy = [y,2*y,4*y]
# yy = np.array(yy).T
# dat.save({'time': x, 'ee_xyz': yy, 'filter': yy},
#         save_location='test_group/diff_test/session000/run000',
#         create=True, overwrite=True)

ax1 = plt.subplot2grid((18,9), (0,0), colspan=9, rowspan=3)
ax21 = plt.subplot2grid((18,9), (3,0), colspan=3, rowspan=3)
ax22 = plt.subplot2grid((18,9), (3,3), colspan=3, rowspan=3)
ax23 = plt.subplot2grid((18,9), (3,6), colspan=3, rowspan=3)
ax31 = plt.subplot2grid((18,9), (6,0), colspan=2, rowspan=3)
ax32 = plt.subplot2grid((18,9), (6,2), colspan=2, rowspan=3)
ax33 = plt.subplot2grid((18,9), (6,4), colspan=2, rowspan=3)
ax61 = plt.subplot2grid((18,9), (6,6), colspan=3, rowspan=3)
ax41 = plt.subplot2grid((18,9), (9,0), colspan=2, rowspan=3)
ax42 = plt.subplot2grid((18,9), (9,2), colspan=2, rowspan=3)
ax43 = plt.subplot2grid((18,9), (9,4), colspan=2, rowspan=3)
ax62 = plt.subplot2grid((18,9), (9,6), colspan=3, rowspan=3)
ax51 = plt.subplot2grid((18,9), (12,0), colspan=2, rowspan=3)
ax52 = plt.subplot2grid((18,9), (12,2), colspan=2, rowspan=3)
ax53 = plt.subplot2grid((18,9), (12,4), colspan=2, rowspan=3)
ax63 = plt.subplot2grid((18,9), (12,6), colspan=3, rowspan=3)


traj = TrajectoryErrorProc(db_name='example_db')
loc = 'friction_post_tuning/nengo_cpu_friction_25_0/session000/run000'
#loc = 'test_group/diff_test/session000/run000'
colors = ['r', 'g', 'b', 'y', 'c']
# -----------test1
data = traj.generate(save_location=loc,
              time_derivative=0,
              filter_const=None,
              interpolated_samples=100,
              clear_memory=True)
raw = dat.load(params=['ee_xyz','filter', 'time'], save_location=loc)
ax1.set_title('Trajectory Error Testing')
ax1.set_ylabel('interpolation_test')
ax1.plot(np.cumsum(raw['time']), raw['ee_xyz'], 'ro', label='raw ee_xyz vs time')
ax1.plot(data['time'], data['ee_xyz'], 'bo', label='proc ee_xyz vs time')
# ax1.plot(np.cumsum(x),yy[:,0], 'ko', label='X')
# ax1.plot(np.cumsum(x),yy[:,1], 'ko', label='Y')
# ax1.plot(np.cumsum(x),yy[:,2], 'ko', label='Z')
ax1.legend()
print('raw: ', np.array(raw['ee_xyz']).shape)
print('proc: ', np.array(data['ee_xyz']).shape)

# -----------test2
ax21.set_ylabel('diff_test')
for order in range(0,3):
    data = traj.generate(save_location=loc,
                  time_derivative=order,
                  filter_const=None,
                  interpolated_samples=100,
                  clear_memory=True)
    ax21.plot(data['time'], data['ee_xyz'][:,0], colors[order+1], label=order)
    ax22.plot(data['time'], data['ee_xyz'][:,1], colors[order+1], label=order)
    ax23.plot(data['time'], data['ee_xyz'][:,2], colors[order+1], label=order)
    ax21.legend()

# -----------test3
ax31.set_ylabel('filter_test diff 0')
alpha = [None, 0.2, 0.5, 0.8]
for ii, a in enumerate(alpha):
    print(a)
    data = traj.generate(save_location=loc,
                  time_derivative=0,
                  filter_const=a,
                  interpolated_samples=100,
                  clear_memory=True)
    if ii == 0:
        alpha[ii] = 'None'
    ax31.plot(data['time'], data['ee_xyz'][:,0], colors[ii], label=alpha[ii])
    ax32.plot(data['time'], data['ee_xyz'][:,1], colors[ii], label=alpha[ii])
    ax33.plot(data['time'], data['ee_xyz'][:,2], colors[ii], label=alpha[ii])
    ax31.legend()
    if ii == 0:
        # -----------test6
        ax61.set_ylabel('error test no filter')
        ax61.plot(data['time'], np.ones(len(data['time']))*data['error'], colors[0], label='2norm error')
        ax61.legend()

# -----------test4
ax41.set_ylabel('filter_test diff 1')
alpha = [None, 0.2, 0.5, 0.8]
for ii, a in enumerate(alpha):
    print(a)
    data = traj.generate(save_location=loc,
                  time_derivative=1,
                  filter_const=a,
                  interpolated_samples=100,
                  clear_memory=True)
    if ii == 0:
        alpha[ii] = 'None'
    ax41.plot(data['time'], data['ee_xyz'][:,0], colors[ii], label=alpha[ii])
    ax42.plot(data['time'], data['ee_xyz'][:,1], colors[ii], label=alpha[ii])
    ax43.plot(data['time'], data['ee_xyz'][:,2], colors[ii], label=alpha[ii])
    ax41.legend()
    if ii == 0:
        # -----------test6
        ax62.set_ylabel('error test no filter')
        ax62.plot(data['time'], np.ones(len(data['time']))*data['error'], colors[0], label='2norm error')
        ax62.legend()


# -----------test5
ax51.set_ylabel('filter_test diff 2')
alpha = [None, 0.2, 0.5, 0.8]
for ii, a in enumerate(alpha):
    print(a)
    data = traj.generate(save_location=loc,
                  time_derivative=2,
                  filter_const=a,
                  interpolated_samples=100,
                  clear_memory=True)
    if ii == 0:
        alpha[ii] = 'None'
    ax51.plot(data['time'], data['ee_xyz'][:,0], colors[ii], label=alpha[ii])
    ax52.plot(data['time'], data['ee_xyz'][:,1], colors[ii], label=alpha[ii])
    ax53.plot(data['time'], data['ee_xyz'][:,2], colors[ii], label=alpha[ii])
    ax51.legend()
    if ii == 0:
        # -----------test6
        ax63.set_ylabel('error test no filter')
        ax63.plot(data['time'], np.ones(len(data['time']))*data['error'], colors[0], label='2norm error')
        ax63.legend()




plt.show()
