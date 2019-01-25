from abr_analyze.utils import (TrajectoryErrorProc, TrajectoryErrorVis,
    DataHandler, DataProcessor)
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

dat = DataHandler(db_name=db_name)
proc = DataProcessor()
trajectory_proc = TrajectoryErrorProc(db_name=db_name)

# variables for progress printout
step = 0
total_steps = len(test_list) * sessions * runs

plt.figure()
ax = plt.subplot(111)
colors = ['r', 'b', 'g', 'y']
# loop through our test list
for ii, test in enumerate(test_list):
    # loop through each session
    session_errors = []
    for session in range(0,sessions):
        errors = []
        # loop through each test
        for run in range(0,runs):
            step += 1
            print('%.2f%% complete'%(step/total_steps*100), end='\r')
            # set the save location for the current run
            save_location = '%s/%s/session%03d/run%03d'%(test_group,
                    test, session, run)
            # load and process our raw data
            data = trajectory_proc.generate(save_location=save_location,
                    time_derivative=time_derivative, filter_const=filter_const,
                    interpolated_samples=interpolated_samples,
                    clear_memory=False)
            # save the trajectory error for the current run
            errors.append(data['error'])
        # save the list of errors over the entire session
        dat.save({'two_norm_error': errors},
                save_location='%s/%s/session%03d'%(test_group, test,
                    session), overwrite=True)
        session_errors.append(errors)
    # calculate the mean and confidence intervals of our mean run errors over
    # each session
    ci_errors = proc.get_mean_and_ci(raw_data=session_errors)
    dat.save(ci_errors, save_location='%s/%s/proc_data'%(test_group, test),
            overwrite=True, create=True)
    print('%s/%s complete.'%(test_group, test))

    ax.fill_between(range(np.array(ci_errors['mean']).shape[0]),
                     ci_errors['upper_bound'],
                     ci_errors['lower_bound'],
                     color=colors[ii],
                     alpha=.5)
    ax.plot(ci_errors['mean'], color=colors[ii])

plt.show()
