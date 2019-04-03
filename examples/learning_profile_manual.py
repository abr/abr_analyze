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
from abr_analyze.nengo_utils import network_utils
from abr_control.controllers import signals
from abr_analyze.paths import cache_dir, figures_dir
from download_examples_db import check_exists as examples_db
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import nengolib

examples_db()
dat = DataHandler('abr_analyze_examples')
fig = plt.figure(figsize=(8,12))
ax_list = [
      fig.add_subplot(311),
      fig.add_subplot(312),
      fig.add_subplot(313)
     ]

runs = 10
for ii in range(0, runs):
    data = dat.load(parameters=['input_signal'],
            save_location='test_1/session000/run%03d'%ii)
    if ii == 0:
        input_signal = data['input_signal']
    else:
        input_signal = np.vstack((input_signal, data['input_signal']))

input_signal = np.squeeze(input_signal)

# specify our network parameters
backend = 'nengo_cpu'
seed = 0
neuron_type = 'lif'
n_neurons = 1000
n_ensembles = 1
test_name = 'learning_profile_example',
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
encoders = network_utils.generate_encoders(
    input_signal=input_signal,
    n_neurons=n_neurons*n_ensembles)

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
network_utils.gen_learning_profile(
    network=network,
    input_signal=input_signal,
    ax_list=ax_list,
    n_ens_to_raster=1,
    show_plot=False)

loc = '%s/examples/learning_profile_manual'%figures_dir
plt.savefig(loc)
print('Figure saved to %s'%loc)
plt.show()
