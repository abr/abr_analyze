"""
"""
from abr_analyze.utils import NetworkUtils, InterceptsScan
from abr_control.controllers import signals
from abr_analyze.utils.paths import cache_dir
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import nengolib

net_utils = NetworkUtils()
scan = InterceptsScan()

# get our saved q and dq values
input_signal_file = 'cpu_53_input_signal.npz'
data = np.load(input_signal_file)
qs = data['qs']
dqs = data['dqs']

# scale q-dq and convert to spherical, but only for the adapting joints
adapt_input = np.array([True, True, True, True, True, False], dtype=bool)
in_index = np.arange(6)[adapt_input]
[qs, dqs] = net_utils.generate_scaled_inputs(q=qs, dq=dqs, in_index=in_index)
input_signal = np.hstack((qs, dqs))

# set the decimal percent from the end of the run to use for input
# 1 == all runs, 0.1 = last 10% of runs
# portion=0.2
# print('Original Input Signal Shape: ', np.array(input_signal).shape)
# input_signal = input_signal[-int(np.array(input_signal).shape[0]*portion):, :]
# print('Input Signal Shape from Selection: ', np.array(input_signal).shape)
input_signal = net_utils.convert_to_spherical(input_signal)

# specify our network parameters
backend = 'nengo_cpu'
seed = 0
neuron_type = 'lif'
n_neurons = 1000
n_ensembles = 20
test_name = '1k x %i: %s'%(n_ensembles, input_signal_file)
n_input = 11
n_output = 5
seed = 0

# ----------- Create your encoders ---------------
encoders = net_utils.generate_encoders(input_signal=input_signal,
        n_neurons=n_neurons*n_ensembles, thresh=0.02)

encoders = encoders.reshape(n_ensembles, n_neurons, n_input)

# ----------- Set input signal to only use portion of data, but still use the
# entire list for encoder selection
# set the decimal percent from the end of the run to use for input
# 1 == all runs, 0.1 = last 10% of runs
portion=0.1
print('Original Input Signal Shape: ', np.array(input_signal).shape)
input_signal = input_signal[-int(np.array(input_signal).shape[0]*portion):, :]
print('Input Signal Shape from Selection: ', np.array(input_signal).shape)


# ----------- generate possible intercept bounds and modes ---------------
intercept_vals = net_utils.gen_intercept_bounds_and_modes()

# ----------- create your ideal profile function ---------------
# triangular distribution
# ideal_function = lambda x: -(100/.6)*x + 100 if x<0.6 else 0
# gaussian distribution
# A == y peak, dependant on how many neurons in your network
# mu == x offset
# sig == std dev
ideal_function = lambda x, A=400, mu=0.4, sig=0.025: A*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# ----------- Instantiate your nengo simulator ---------------
# This example uses the network defined in
# abr_control/controllers/signals/dynamics_adaptation.py
scan.proportion_time_active(
    n_input=n_input,
    n_output=n_output,
    n_neurons=n_neurons,
    n_ensembles=n_ensembles,
    pes_learning_rate=1e-6,
    intercept_vals=intercept_vals,
    backend=backend,
    seed=seed,
    neuron_type=neuron_type,
    encoders=encoders,
    input_signal=input_signal,
    ideal_function=ideal_function)
