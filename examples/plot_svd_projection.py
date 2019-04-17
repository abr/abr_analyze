"""
An example using the 3 SVD projection functions in data_visualizer.py.
The input signal is projected into a 1D, 2D, and 3D space, and plotted
against the first dimension of u_adapt.
"""
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from abr_analyze import DataHandler
import abr_analyze.data_visualizer as vis
from abr_analyze.plotting import TrajectoryError
from abr_analyze.paths import figures_dir
from download_examples_db import check_exists as examples_db


examples_db()
dat = DataHandler('abr_analyze_examples')

data = dat.load(parameters=['input_signal', 'u_adapt'],
                save_location='test_1/session000/run000')
input_signal = data['input_signal'].squeeze()
output_signal = data['u_adapt'][:, 0].squeeze()

fig, axes = plt.subplots(3, 1)

vis.plot_against_projection_2d(
    ax=axes[0], data_project=input_signal, data_plot=output_signal)
vis.plot_against_projection_3d(
    ax=axes[1], data_project=input_signal, data_plot=output_signal)
vis.plot_against_projection_4d(
    ax=axes[2], data_project=input_signal, data_plot=output_signal)

fig.suptitle('SVD projection')
fig.tight_layout()
fig.subplots_adjust(top=0.88)

loc = '%s/svd_projection.png'%(figures_dir)
print('Figure saved to %s'%(loc))
plt.savefig(loc)
plt.show()
