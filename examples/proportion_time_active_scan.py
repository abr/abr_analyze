"""
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
from nengolib.stats import ScatteredHypersphere

# get our saved q and dq values
input_signal_file = 'cpu_53_input_signal.npz'
data = np.load(input_signal_file)
qs = data['qs']
dqs = data['dqs']

# scale q-dq and convert to spherical, but only for the adapting joints
adapt_input = np.array([True, True, True, True, True, False], dtype=bool)
in_index = np.arange(6)[adapt_input]
qs, dqs = network_utils.generate_scaled_inputs(q=qs, dq=dqs, in_index=in_index)

input_signal = np.hstack((qs, dqs))

# set the decimal percent from the end of the run to use for input
# 1 == all runs, 0.1 = last 10% of runs
# portion=0.2
# print('Original Input Signal Shape: ', np.array(input_signal).shape)
# input_signal = input_signal[-int(np.array(input_signal).shape[0]*portion):, :]
# print('Input Signal Shape from Selection: ', np.array(input_signal).shape)
input_signal = nengolib.stats.spherical_transform(input_signal)

# specify our network parameters
seed = 0
n_neurons = 1000
n_ensembles = 10
test_name = '1k x %i: %s'%(n_ensembles, input_signal_file)
n_input = 11

# ----------- Create your encoders ---------------
hypersphere = ScatteredHypersphere(surface=True)
encoders = hypersphere.sample(n_neurons*n_ensembles, n_input)

encoders = encoders.reshape(n_ensembles, n_neurons, n_input)

# ----------- Set input signal to only use portion of data, but still use the
# entire list for encoder selection
# set the decimal percent from the end of the run to use for input
# 1 == all runs, 0.1 = last 10% of runs
portion=1
print('Original Input Signal Shape: ', np.array(input_signal).shape)
input_signal = input_signal[-int(np.array(input_signal).shape[0]*portion):, :]
print('Input Signal Shape from Selection: ', np.array(input_signal).shape)


# ----------- generate possible intercept bounds and modes ---------------
intercept_vals = network_utils.gen_intercept_bounds_and_modes(
    intercept_step=0.2, mode_step=0.2)

# ----------- Instantiate your nengo simulator ---------------
# This example uses the network defined in
# abr_control/controllers/signals/dynamics_adaptation.py
save_name = 'proportion_time'
intercepts_scan.proportion_time_active(
    encoders=encoders,
    intercept_vals=intercept_vals,
    input_signal=input_signal,
    seed=seed,
    save_name=save_name,
    pes_learning_rate=1e-6,
    notes=
    '''
    - 10k ensemble size
    - encoders selected from scattered hypersphere
    ''',
    )

# ----------- create your ideal profile function ---------------
# triangular distribution
# ideal_function = lambda x: -(100/.6)*x + 100 if x<0.6 else 0
# gaussian distribution
# A == y peak, dependent on how many neurons in your network
# mu == x offset
# sig == std dev
def gauss(x):
    return 400*np.exp(-np.power(x - 0.4, 2.) / (2 * 0.025**2))

# ----- compare generated data to ideal, plot 10 closest ----
intercepts_scan.review(
    save_name=save_name,
    ideal_function=gauss,
    num_to_plot=10,
    )
