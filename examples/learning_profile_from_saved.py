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
from abr_control.controllers import signals
from download_examples_db import check_exists as examples_db

from abr_analyze import DataHandler
from abr_analyze.nengo import network_utils
from abr_analyze.paths import cache_dir, figures_dir

examples_db()
dat = DataHandler("abr_analyze_examples")
fig = plt.figure(figsize=(8, 12))
ax_list = [fig.add_subplot(311), fig.add_subplot(312), fig.add_subplot(313)]
data = dat.load(
    parameters=[
        "n_input",
        "n_output",
        "n_neurons",
        "n_ensembles",
        "pes",
        "intercepts",
        "seed",
        "encoders",
    ],
    save_location="nengo_data",
)

n_input = int(data["n_input"])
n_output = int(data["n_output"])
n_neurons = int(data["n_neurons"])
n_ensembles = int(data["n_ensembles"])
pes_learning_rate = float(data["pes"])
intercepts = data["intercepts"]
intercepts = np.array(intercepts)
intercepts = intercepts.reshape((n_ensembles, n_neurons))
seed = int(data["seed"])
encoders = data["encoders"]

runs = 10
for ii in range(0, runs):
    data = dat.load(
        parameters=["input_signal"], save_location="test_1/session000/run%03d" % ii
    )
    if ii == 0:
        input_signal = data["input_signal"]
    else:
        input_signal = np.vstack((input_signal, data["input_signal"]))

input_signal = np.squeeze(input_signal)

network = signals.DynamicsAdaptation(
    n_input=n_input,
    n_output=n_output,
    n_neurons=n_neurons,
    n_ensembles=n_ensembles,
    pes_learning_rate=1e-6,
    intercepts=intercepts,
    seed=seed,
    encoders=encoders,
)

network_utils.gen_learning_profile(
    network=network,
    input_signal=input_signal,
    ax_list=ax_list,
    n_ens_to_raster=1,
    show_plot=False,
)

loc = "%s/learning_profile_from_saved" % figures_dir
plt.savefig(loc)
print("Figure saved to %s" % loc)
plt.show()
