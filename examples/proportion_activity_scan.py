"""
Run a parameter sweep across intercept values, looking at the proportion
of neurons are active over time and the proportion of time neurons are active.
Display the 10 results that are closest to the ideal function specified
"""
from abr_analyze.nengo import network_utils, intercepts_scan
from abr_control.controllers import signals
from abr_analyze.paths import cache_dir
from abr_analyze import DataHandler
from download_examples_db import check_exists as examples_db
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

examples_db()
runs = 10
dat = DataHandler('abr_analyze_examples')
for ii in range(0, runs):
    data = dat.load(parameters=['input_signal'],
            save_location='test_1/session000/run%03d'%ii)
    if ii == 0:
        input_signal = data['input_signal']
    else:
        input_signal = np.vstack((input_signal, data['input_signal']))

input_signal = np.squeeze(input_signal)

# specify our network parameters
seed = 0
n_neurons = 1000
n_ensembles = 1
n_input = 11

# ----------- Create your encoders ---------------
encoders = network_utils.generate_encoders(
    input_signal=input_signal,
    n_neurons=n_neurons*n_ensembles,
    thresh=0.01)

encoders = encoders.reshape(n_ensembles, n_neurons, n_input)

# ----------- Set input signal to only use portion of data, but still use the
# entire list for encoder selection
# set the decimal percent from the end of the run to use for input
# 1 == all runs, 0.1 = last 10% of runs
portion=0.1
print('Original input signal shape: ', np.array(input_signal).shape)
input_signal = input_signal[-int(np.array(input_signal).shape[0]*portion):, :]
print('Input signal shape from selection: ', np.array(input_signal).shape)


# ----------- generate possible intercept bounds and modes ---------------
intercept_vals = network_utils.gen_intercept_bounds_and_modes(
    intercept_step=0.5, mode_step=0.5)
print('%i different combinations to be tested' % len(intercept_vals))

# ----------- Instantiate your nengo simulator ---------------
# This example uses the network defined in
# abr_control/controllers/signals/dynamics_adaptation.py
save_name = 'proportion_activity'
analysis_fncs = [
    network_utils.proportion_neurons_active_over_time,
    network_utils.proportion_time_neurons_active
    ]
intercepts_scan.run(
    encoders=encoders,
    intercept_vals=intercept_vals,
    input_signal=input_signal,
    seed=seed,
    save_name=save_name,
    pes_learning_rate=1e-6,
    notes='',
    analysis_fncs=analysis_fncs)

# ----- compare generated data to ideal, plot 10 closest ----
save_name0 = '%s/%s'%(save_name, analysis_fncs[0].__name__)
intercepts_scan.review(
    save_name=save_name0,
    ideal_function=lambda x: 0.3,
    num_to_plot=10,
    )

def gauss(x, a=400, mu=0.4, sig=0.025):
    return a*np.exp(-np.power(x - mu, 2.) / (2 * sig**2))
save_name1 = '%s/%s'%(save_name, analysis_fncs[1].__name__)
intercepts_scan.review(
    save_name=save_name1,
    ideal_function=gauss,
    num_to_plot=10,
    )

plt.show()
