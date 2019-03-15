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

test_group = 'friction_post_tuning'
test_names = [
        'nengo_cpu_friction_103_0',
        #'nengo_cpu_friction_67_0'
        ]
runs = 50
db_name = 'dewolf2018neuromorphic'
dat = DataHandler(db_name=db_name)

fig = plt.figure(figsize=(8,12))
ax = [
      fig.add_subplot(311),
      fig.add_subplot(312),
      fig.add_subplot(313)
     ]
for test_name in test_names:
    loc = '/%s/%s/'%(test_group, test_name)
    data = dat.load(parameters=['n_input', 'n_output', 'n_neurons', 'n_ensembles',
        'pes', 'intercepts', 'backend', 'seed', 'neuron_type', 'encoders'],
        save_location='%sparameters/dynamics_adaptation'%loc)

    n_input = int(data['n_input'])
    n_output = int(data['n_output'])
    n_neurons = int(data['n_neurons'])
    n_ensembles = int(data['n_ensembles'])
    pes_learning_rate = float(data['pes'])
    intercepts = data['intercepts']
    backend = data['backend'].tolist()
    seed = int(data['seed'])
    neuron_type = data['neuron_type'].tolist()
    encoders = data['encoders']

    for ii in range(0, runs):
        data = dat.load(parameters=['input_signal'],
                save_location='%s/session000/run%03d'%(loc,ii))
        if ii == 0:
            input_signal = data['input_signal']
        else:
            input_signal = np.vstack((input_signal, data['input_signal']))

    input_signal = np.squeeze(input_signal)

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

    net_utils.gen_learning_profile(
        network=network,
        input_signal=input_signal,
        ax=ax,
        num_ens_to_raster=1,
        show_plot=False)
plt.show()
