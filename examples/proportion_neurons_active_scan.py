"""
Run a parameter sweep across intercept values, looking at the proportion
of neurons are active over time. Display the 10 results that are closest
to the ideal function specified (y=0.3)
"""
from abr_analyze.nengo_utils import network_utils, intercepts_scan
from abr_control.controllers import signals
from abr_analyze.paths import cache_dir
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import nengolib

# get our saved q and dq values
input_signal_file = 'cpu_53_input_signal.npz'
data = np.load(input_signal_file)
qs = data['qs']
dqs = data['dqs']

# scale q-dq and convert to spherical, but only for the adapting joints
adapt_input = np.array([True, True, True, True, True, False], dtype=bool)
in_index = np.arange(6)[adapt_input]
[qs, dqs] = network_utils.generate_scaled_inputs(q=qs, dq=dqs, in_index=in_index)
input_signal = np.hstack((qs, dqs))

input_signal = nengolib.stats.spherical_transform(input_signal)

# specify our network parameters
seed = 0
n_neurons = 1000
n_ensembles = 15
test_name = '1k x %i: %s'%(n_ensembles, input_signal_file)
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
save_name = 'proportion_neurons'
intercepts_scan.proportion_neurons_active(
    encoders=encoders,
    intercept_vals=intercept_vals,
    input_signal=input_signal,
    seed=seed,
    save_name=save_name,
    pes_learning_rate=1e-6,
    notes='',
    )

# ----- compare generated data to ideal, plot 10 closest ----
intercepts_scan.review(
    save_name=save_name,
    ideal_function=lambda x: 0.3,
    num_to_plot=10,
    )
plt.show()
