import pytest
import numpy as np

import nengo

from abr_analyze.nengo_utils import network_utils

class DynamicsAdaptation:
    def __init__(self, n_neurons, n_ensembles, **kwargs):
        self.n_neurons = n_neurons
        self.n_ensembles = n_ensembles
        self.input_signal = 0
        self.tau_output = 0.2

        self.nengo_model = nengo.Network()
        with self.nengo_model:

            def input_func(t):
                return self.input_signal
            input_node = nengo.Node(input_func, size_out=1)

            self.adapt_ens = []
            self.probe_neurons = []
            for ii in range(n_ensembles):
                # create ensemble
                self.adapt_ens.append(nengo.Ensemble(
                    n_neurons, dimensions=1, **kwargs))
                # connect to input
                nengo.Connection(input_node, self.adapt_ens[ii])
                # create neuron activity probe
                self.probe_neurons.append(
                    nengo.Probe(self.adapt_ens[-1].neurons, synapse=None))

        self.sim = nengo.Simulator(self.nengo_model)

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
def test_raster_plot(network, num_ens_to_raster, plt):

    input_signal = np.sin(np.linspace(0, 2*np.pi, 100))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    network_utils.raster_plot(network, input_signal, ax, num_ens_to_raster)


@pytest.mark.parametrize('network, input_signal, answer', (
    (DynamicsAdaptation(1, 1, encoders=[[1]], max_rates=[10]),
     np.ones(1000), 10),
    (DynamicsAdaptation(1, 1, encoders=[[1]], max_rates=[100]),
     np.ones(1000), 100),
    (DynamicsAdaptation(1, 1, encoders=[[-1]], max_rates=[100]),
     -1 * np.ones(1000), 100),
    (DynamicsAdaptation(1, 1, encoders=[[1]], max_rates=[100]),
     -1 * np.ones(1000), 0),
    ))
def test_get_activities(network, input_signal, answer):

    dt = 0.001
    spike_trains = network_utils.get_activities(
        network=network,
        input_signal=np.array(input_signal),
        dt=dt,
        synapse=None)

    # allowable error is 2.5% of max firing rate
    threshold = np.ceil(network.adapt_ens[0].max_rates[0] * 0.025)
    # assert the result is within 1 of the expected spiking rate
    assert abs(np.sum(spike_trains) - answer) <= threshold


# expected sum of proportion of total neurons active over time over 1 second
# is n_neurons_active / n_neurons * max_rates
@pytest.mark.parametrize('network, input_signal, answer', (
    (DynamicsAdaptation(2, 1, encoders=[[1], [1]], max_rates=[10, 10]),
     np.ones((1000, 1)), 2 / 2 * 10),
    (DynamicsAdaptation(2, 1, encoders=[[1], [1]], max_rates=[10, 100]),
     np.ones((1000, 1)), 1 / 2 * 10 + 1 / 2 * 100),
    (DynamicsAdaptation(2, 1, encoders=[[1], [-1]], max_rates=[10, 10]),
     np.ones((1000, 1)), 1 / 2 * 10),
    (DynamicsAdaptation(2, 1, encoders=[[-1], [-1]], max_rates=[10, 10]),
     np.ones((1000, 1)), 0 / 2 * 10),
    (DynamicsAdaptation(1, 1, encoders=[[1]], max_rates=[100]),
     np.ones((1000, 1)), 1 / 1 * 100),
    ))
def test_proportion_neurons_active_over_time(network, input_signal, answer, plt):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    proportion_neurons_active, _ = (
        network_utils.proportion_neurons_active_over_time(
            input_signal=input_signal, network=network, synapse=None, ax=ax))

    threshold = np.ceil(network.adapt_ens[0].max_rates[0] * 0.025)
    print('threshold: ', threshold)
    assert abs(np.sum(proportion_neurons_active) - answer) <= threshold


# expected sum of the proportion of time neurons are active over 1 second
# (assuming input signal is always in the preferred direction) is
# n_neurons * max_firing_rates / timesteps
@pytest.mark.parametrize('network, input_signal, answer', (
    (DynamicsAdaptation(2, 1, encoders=[[1], [1]], max_rates=[10, 10]),
     np.ones((1000, 1)), 2 * 10 / 1000),
    (DynamicsAdaptation(2, 1, encoders=[[1], [1]], max_rates=[10, 100]),
     np.ones((1000, 1)), 1 * 10 / 1000 + 1 * 100 / 1000),
    (DynamicsAdaptation(1, 1, encoders=[[1]], max_rates=[100]),
     np.ones((1000, 1)), 1 * 100 / 1000),
    (DynamicsAdaptation(100, 1, encoders=[[1]]*100, max_rates=[100]*100),
     np.ones((1000, 1)), 100 * 100 / 1000),
    ))
def test_proportion_time_neurons_active(network, input_signal, answer, plt):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    proportion_time_active, _ = network_utils.proportion_time_neurons_active(
        input_signal=input_signal, network=network, synapse=None, ax=ax)

    # allowable error is 2.5% of max firing rate / s * # of neurons
    threshold = (np.ceil(network.adapt_ens[0].max_rates[0] * 0.03) /
                 input_signal.shape[0] * network.n_neurons)
    assert round(abs(np.sum(proportion_time_active) - answer), 6) <= threshold


@pytest.mark.parametrize('network, input_signal, answer', (
    (DynamicsAdaptation(2, 1, encoders=[[1], [1]], max_rates=[10, 10]),
     np.ones((1000, 1)), (2, 0)),
    (DynamicsAdaptation(2, 1, encoders=[[1], [1]], max_rates=[10, 100]),
     np.ones((1000, 1)), (2, 0)),
    (DynamicsAdaptation(2, 1, encoders=[[1], [-1]], max_rates=[10, 10]),
     np.ones((1000, 1)), (1, 1)),
    (DynamicsAdaptation(100, 1, encoders=[[1],[-1]]*50, max_rates=[100]*100),
     np.ones((1000, 1)), (50, 50))
    ))
def test_n_neurons_active_and_inactive(network, input_signal, answer):

    spike_train = network_utils.get_activities(
        network=network, input_signal=np.array(input_signal), synapse=None)

    n_active, n_inactive = network_utils.n_neurons_active_and_inactive(
        spike_train)

    assert n_active == answer[0]
    assert n_inactive == answer[1]


def test_gen_learning_profile(plt):

    # test without ax provided
    network=DynamicsAdaptation(1, 1, encoders=[[1]], max_rates=[10])
    input_signal=np.ones((1000, 1))
    network_utils.gen_learning_profile(
        network=network,
        input_signal=input_signal,
        show_plot=False)
    # and with ax provided
    fig = plt.figure()
    ax_list = [fig.add_subplot(3, 1, ii+1) for ii in range(3)]
    network_utils.gen_learning_profile(
        network, input_signal, ax_list=ax_list, show_plot=False)


@pytest.mark.parametrize(
    'intercept_range, intercept_step, mode_range, mode_step', (
        (None, 0.1, None, 0.2),
        (None, 0.05, None, 0.01),
        (None, 0.5, None, 0.7),
        )
    )
def test_gen_intercept_bounds_and_modes(intercept_range, intercept_step,
                                        mode_range, mode_step):


    intercepts = network_utils.gen_intercept_bounds_and_modes(
        intercept_range, intercept_step, mode_range, mode_step)

    assert (np.all(intercepts[:, 0] <= intercepts[:, 2]) and
            np.all(intercepts[:, 2] <= intercepts[:, 1]))
