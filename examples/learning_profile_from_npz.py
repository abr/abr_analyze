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
import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import os

import matplotlib.pyplot as plt
from nengo_extras import dists

from abr_analyze import DataHandler
from abr_analyze.nengo import network_utils
from abr_analyze.paths import cache_dir, figures_dir
from abr_control._vendor.nengolib.stats import ScatteredHypersphere
from abr_control.controllers import signals

npz = "run_0_obj_53.npz"
data = np.load(npz)

# specify our network parameters
backend = "nengo_cpu"
seed = int(data["seed"])
n_neurons = 3000
n_ensembles = 1
test_name = ("learning_profile_l2m",)
n_input = 13
n_output = 6
pes = data["learning_rate"]
input_signal = data["input_signal"]

np.random.RandomState(seed)
# ----------- Create your intercepts ---------------
intercepts = dists.generate_triangular(
    n_input=n_input,
    n_ensembles=n_ensembles,
    n_neurons=n_neurons,
    bounds=[-0.1, 0.1],
    mode=0.0,
    seed=seed,
)

# ----------- Create your encoders ---------------
hypersphere = ScatteredHypersphere(surface=True)
encoders = hypersphere.sample(n_ensembles * n_neurons, n_input)
encoders = encoders.reshape(n_ensembles, n_neurons, n_input)

# ----------- Instantiate your nengo simulator ---------------
# This example uses the network defined in
# abr_control/controllers/signals/dynamics_adaptation.py
network = signals.DynamicsAdaptation(
    n_input=n_input,
    n_output=n_output,
    n_neurons=n_neurons,
    n_ensembles=n_ensembles,
    pes_learning_rate=pes,
    intercepts=intercepts,
    seed=seed,
    encoders=encoders,
)

# create our figure object
fig = plt.figure(figsize=(8, 12))
ax_list = [fig.add_subplot(311), fig.add_subplot(312), fig.add_subplot(313)]
plt.tight_layout()

# pass your network and input signal to the network utils module
# run a sim and plot the learning profile
network_utils.gen_learning_profile(
    network=network,
    input_signal=input_signal,
    ax_list=ax_list,
    n_ens_to_raster=1,
    show_plot=False,
)

loc = "%s/learning_profile_l2m" % figures_dir
plt.savefig(loc)
print("Figure saved to %s" % loc)
plt.show()
