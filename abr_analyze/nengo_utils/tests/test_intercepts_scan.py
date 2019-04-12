import pytest
import numpy as np

from abr_analyze.nengo_utils import intercepts_scan, network_utils

def get_params():
    n_ensembles = 1
    n_neurons = 50
    # encoders should be n_ensembles x n_neurons x n_dims
    encoders = np.array([np.random.choice([-1, 1], n_neurons)[:, None]
                         for ii in range(n_ensembles)])
    intercept_vals = [
        [-.5, 0, -.25],
        [-.9, .5, .25],
        [.5, 1, .75],
        ]
    input_signal = np.sin(np.linspace(0, 2*np.pi, 100))

    return encoders, intercept_vals, input_signal



def test_run():
    encoders, intercept_vals, input_signal = get_params()

    intercepts_scan.run(
        encoders=encoders,
        intercept_vals=intercept_vals,
        input_signal=input_signal,
        analysis_fncs=[network_utils.proportion_time_neurons_active,
                       network_utils.proportion_neurons_active_over_time]
        )


def test_review(plt):
    encoders, intercept_vals, input_signal = get_params()

    save_name_pna = 'proportion_neurons_active'
    analysis_fncs = network_utils.proportion_neurons_active_over_time
    review_name='%s/%s'%(save_name_pna, analysis_fncs.__name__)

    intercepts_scan.run(
        encoders=encoders,
        intercept_vals=intercept_vals,
        input_signal=input_signal,
        save_name=save_name_pna,
        analysis_fncs=[analysis_fncs]
        )

    intercepts_scan.review(
        save_name=review_name,
        ideal_function=lambda x: 0.3,
        num_to_plot=3
        )
    plt.show()
