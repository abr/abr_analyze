"""
Runs a simulation using the data recorded from a run of learning to get
information about the neural network
Plots will be created for each reach, or the inputs can be stacked to get
activity over the whole reaching space

The intercepts bounds and mode can be altered to see how it would affect the
network provided the same input. This helps tune your network without having to
rerun tests

Plots
1. rasterplot showing spikes for each neuron over time
2. proportion of time active, the number of neurons active for what proportion
   of run time
3. proportion of neurons that are active over time
"""
from abr_analyze import DataHandler
from abr_analyze.nengo_utils import NetworkUtils
from abr_control.controllers import signals
from abr_analyze.paths import cache_dir
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import nengolib

net_utils = NetworkUtils()

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
portion=0.2
print('Original Input Signal Shape: ', np.array(input_signal).shape)
input_signal = input_signal[-int(np.array(input_signal).shape[0]*portion):, :]
print('Input Signal Shape from Selection: ', np.array(input_signal).shape)
input_signal = net_utils.convert_to_spherical(input_signal)

# specify our network parameters
backend = 'nengo_cpu'
seed = 0
neuron_type = 'lif'
n_neurons = 1000
n_ensembles = 1
test_name = '1k x %i: %s'%(n_ensembles, input_signal_file)
n_input = 11
n_output = 5
seed = 0

# ----------- Create your intercepts ---------------
intercepts = signals.AreaIntercepts(
    dimensions=n_input,
    base=signals.Triangular(-0.5, -0.5, -0.45))

rng = np.random.RandomState(seed)
intercepts = intercepts.sample(n_neurons, rng=rng)
intercepts = np.array(intercepts)

# ----------- Create your encoders ---------------
encoders = net_utils.generate_encoders(input_signal=input_signal,
        n_neurons=n_neurons*n_ensembles, n_dims=n_input)

encoders = encoders.reshape(n_ensembles, n_neurons, n_input)

# ----------- Instantiate your nengo simulator ---------------
# This example uses the network defined in
# abr_control/controllers/signals/dynamics_adaptation.py
network = signals.DynamicsAdaptation(
    n_input=n_input,
    n_output=n_output,
    n_neurons=n_neurons,
    n_ensembles=n_ensembles,
    pes_learning_rate=1e-6,
    intercepts=intercepts,
    backend=backend,
    probe_weights=True,
    seed=seed,
    neuron_type=neuron_type,
    encoders=encoders)

# pass your network and input signal to the network utils module
# run a sim and plot the learning profile
net_utils.gen_learning_profile(
    network=network,
    input_signal=input_signal,
    ax=None,
    num_ens_to_raster=1)
