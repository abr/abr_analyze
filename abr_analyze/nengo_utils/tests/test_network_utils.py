import pytest
import numpy as np

import nengo

from abr_analyze.nengo_utils import network_utils

class DynamicsAdaptation:
    def __init__(self, n_neurons, n_ensembles):
        self.n_neurons = n_neurons
        self.n_ensembles = n_ensembles

        self.nengo_model = nengo.Network()
        with self.nengo_model:

            def input_func(t):
                return self.input_signal
            input_node = nengo.Node(input_func, size_out=1)

            self.adapt_ens = []
            for ii in range(n_ensembles):
                self.adapt_ens.append(nengo.Ensemble(n_neurons, dimensions=1))
                nengo.Connection(input_node, self.adapt_ens[ii])

@pytest.mark.parametrize('thresh', ((0.001, 0.01, 0.1, 1.0)))
def test_generate_encoders(thresh):

    x = np.arange(0, 2*np.pi, 0.001)
    input_signal = np.vstack([np.sin(x), np.cos(x)]).T

    n_neurons = 100
    encoders = network_utils.generate_encoders(
        n_neurons=n_neurons,
        input_signal=input_signal,
        thresh=thresh)

    # make sure there are no points within threshould
    # distance of each other
    for ii in range(n_neurons):
        assert np.all(np.linalg.norm(encoders - encoders[ii]) > thresh)
    print('Encoders: ', encoders)


@pytest.mark.parametrize('q', (np.zeros(6), np.ones(6)*6.28))
@pytest.mark.parametrize('dq', (np.ones(6)*-2.5, np.ones(6)*2.5))
@pytest.mark.parametrize('in_index', (
    [0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]))
def test_generate_scaled_inputs(q, dq, in_index):

    scaled_q, scaled_dq = network_utils.generate_scaled_inputs(
        q, dq, [in_index])

    assert np.all(0 <= scaled_q) and np.all(scaled_q  <= 1)
    assert np.all(0 <= scaled_dq) and np.all(scaled_dq <= 1)


# NOTE: this tests plots are properly generated, for full confirmation
# the plots in results folder should be visually inspected
@pytest.mark.parametrize('network, num_ens_to_raster', (
    (DynamicsAdaptation(10, 1), 1),
    (DynamicsAdaptation(10, 10), 10)),
    )
def test_raster_plot(network, num_ens_to_raster):
    import matplotlib.pyplot as plt

    input_signal = np.sin(np.linspace(0, 2*np.pi, 100))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    network_utils.raster_plot(network, input_signal, ax, num_ens_to_raster)
    plt.savefig('results/test_raster_plot_%i_%i' %
                (network.n_neurons, network.n_ensembles))
