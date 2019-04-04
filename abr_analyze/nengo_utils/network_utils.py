"""
various tools for use with Nengo such as input signal scaling and running nengo
simulations to return the neural profile showing different activity metrics.

The profile can consist of:
- the proportion of active neurons over time
- the proportion of time neurons are active
- raster plot of the activity

These can be run individually for any nengo network, or the parameters to
instantiate a dynamics_adaptation network from abr_control can be passed in to
get_learning_profile() to run all three of the above

NOTE: see examples of running for a single profile, or looping through various
intercepts to later view in the intercept_scan_viewer.py gui
"""

import matplotlib.pyplot as plt
import numpy as np

import nengo
from nengo.utils.matplotlib import rasterplot
from nengolib.stats import ScatteredHypersphere

def generate_encoders(n_neurons, input_signal=None, thresh=0.008, depth=0):
    """
    Accepts an input_signal in the shape of time X dim and outputs encoders
    for the specified number of neurons by sampling from the input.
    The selection is made by choosing inputs randomly and checking that
    they are minimally thresh away from one another to avoid overly similar
    encoders. If we have exhausted the search then we increase thresh and
    rescan, until len(encoders) == number of neurons. If there are not
    enough input signal samples to select for the number of neurons, or an
    input_signal of None is passed in, the remainder will be filled with a
    selection from a scattered hypersphere

    PARAMETERS
    ----------
    n_neurons : int
        the number of neurons in the simulation, needed to get
        a corresponding number of encoders
    input_signal : array(time_steps, dimensions), Optional (Default: None)
        the input signal to sample from
    thresh : float, Optional (Default: 0.008)
        the threshold to keep encoder values minimally apart from
        It may be helpful to increase this value if there are many more
        inputs than there are neurons in the final simulation to speed up
        the encoder selection process
    depth : int
        how many times has this function been recursively called
    """

    # first run so we need to generate encoders for the sessions
    ii = 0
    iters_with_no_update = 0
    prev_n_indices = 0
    while input_signal.shape[0] > n_neurons:

        ii += 1
        if (ii % 1000) == 0:
            print('Downsampled to %i encoders at iteration %i' %
                  (input_signal.shape[0], ii),
                  'Current threshold value: %.3f' % thresh, end='\r')

        # choose a random set of indices
        n_indices = input_signal.shape[0]

        # make sure we're dealing with an even number
        n_indices -= int(n_indices % 2)
        n_half = int(n_indices / 2)

        # split data into two random groups
        randomized_indices = np.random.permutation(range(n_indices))
        a = randomized_indices[:n_half]
        b = randomized_indices[n_half:]
        data1 = input_signal[a]
        data2 = input_signal[b]

        # calculate the 2 norm between random pairs between data1 and 2
        distances = np.linalg.norm(data1 - data2, axis=1)
        # find any pairs within threshold distance of each other
        under_thresh = distances > thresh
        # remove the values from data2 within thresh of corresponding data1
        input_signal = np.vstack([data1, data2[under_thresh]])

        if prev_n_indices == n_indices:
            iters_with_no_update += 1
        else:
            iters_with_no_update = 0

        # if we've run 50 iterations but we're still haven't downsampled
        # enough, increase the threshold distance and keep going
        if iters_with_no_update == 50:
            iters_with_no_update = 0
            thresh += .1 * thresh
            # print('All values within threshold, but not at target size.')
            # print('Increasing threshold to %.4f' % thresh)
        prev_n_indices = n_indices

    if input_signal.shape[0] != n_neurons:
        print('Too many indices removed, appending samples from ' +
              'ScatteredHypersphere')
        length = n_neurons - input_signal.shape[0] + 1
        hypersphere = ScatteredHypersphere(surface=True)
        hyper_inputs = hypersphere.sample(length, input_signal.shape[1])
        input_signal = np.vstack((input_signal, hyper_inputs))

        # make sure that the new inputs meet threshold constraints
        if depth < 10:
            input_signal = generate_encoders(
                n_neurons,
                input_signal,
                thresh=thresh,
                depth=depth+1)
        else:
            # if we've tried several times to find input meeting
            # outside threshold distance but failed, return with warning
            # so we're not stuck in infinite loop
            import warnings
            warnings.warn(
                'Could not find set of encoders outside thresh distance')

    # clear the previous recurrent print
    print('\n')
    return np.array(input_signal)


def generate_scaled_inputs(q, dq, in_index):
    '''
    Currently set to accept joint position and velocities as time
    x dimension arrays, and returns them scaled from 0 to 1

    PARAMETERS
    ----------
    q: array of shape time_steps x dimension
        the joint positions to scale
    dq: array of shape time_steps x dimension
        the joint velocities to scale
    in_index: list of integers
        a list corresponding what joints to return scaled inputs for.
        The function is currently set up to accept the raw feedback from an
        arm (all joint position and velocities) and only returns the values
        specified by the indices in in_index

        EX: in_index = [0, 1, 3, 6]
        will return the scaled joint position and velocities for joints 0,
        1, 3 and 6
    '''
    #TODO: generalize this so the user passes in means and scales? right
    #now this is specific for the jaco
    #NOTE: do we want to have in_index here, or have the user specify what
    # joint positions and velocities they pass in?
    qs = q.T
    dqs = dq.T

    # expected mean of joint angles / velocities
    means_q = np.array([0, 0, 0, 0, 0, 0])
    means_dq = np.array([1.25, 1.25, 1.25, 1.25, 1.25, 1.25])

    # expected variance of joint angles / velocities
    scales_q = np.array([6.28, 6.28, 6.28, 6.28, 6.28, 6.28,])
    scales_dq = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5])

    for pp in range(0, 5):
        qs[pp] = (qs[pp] - means_q[pp]) / scales_q[pp]
        dqs[pp] = (dqs[pp] + means_dq[pp]) / scales_dq[pp]
    qs = np.clip(qs, 0, 1)
    dqs = np.clip(dqs, 0, 1)

    scaled_q = np.array([qs[ii] for ii in in_index]).T
    scaled_dq = np.array([dqs[ii] for ii in in_index]).T

    return [scaled_q, scaled_dq]


def raster_plot(network, input_signal, ax, n_ens_to_raster=None,
                n_neurons_per_ens_to_plot=-1):
    '''
    Accepts a Nengo network and runs a simulation with the input_signal
    Plots rasterplot onto ax object up to n_ens_to_raster ensembles
    if n_ens_to_raster is None, all ensembles will be plotted

    PARAMETERS
    ----------
    network: .DynamicsAdaptation
        'abr_control.controllers.signals.dynamics_adaptation'
    input_signal: np array shape of (time_steps x input_dim)
        the input used for the network sim
    ax: ax object
        used for the rasterplot
    n_ens_to_raster: int, Optional (Default: None)
        the number of ensembles to plot in the raster,
        if None all will be plotted
    n_neurons_per_ens_to_plot: int, Optional (Default: -1)
        the number of neurons to plot per ensemble, -1 will plot all
    '''
    if n_ens_to_raster is None:
        n_ens_to_raster = len(network.adapt_ens)

    spike_trains = get_spike_trains(network, input_signal)

    time = np.ones(len(input_signal))
    ax = rasterplot(np.cumsum(time),
                    spike_trains[:, :n_neurons_per_ens_to_plot],
                    ax=ax)

    ax.set_ylabel('Neuron')
    ax.set_xlabel('Time [sec]')
    ax.set_title('Spiking Activity')

    return spike_trains


def get_activities(network, input_signal):
    '''
    Accepts a Nengo network and input signal and returns a list of the neural
    activities based on the encoders, does not incorporate neural dynamics

    PARAMETERS
    ----------
    network: .DynamicsAdaptation
        'abr_control.controllers.signals.dynamics_adaptation'
    input_signal: np array shape of (time_steps x input_dim)
        the input used for the network sim
    '''

    thresh = 1e-5
    activities = []
    for ens in network.adapt_ens:
        _, activity = nengo.utils.ensemble.tuning_curves(
            ens, network.sim, input_signal)
        activity[activity > thresh] = 1
        activity[activity <= thresh] = 0
        activities.append(np.copy(activity))

    return np.array(activities)


def get_spike_trains(network, input_signal, dt=0.001, synapse=None):
    '''
    Accepts a Nengo network and input signal and simulates it, returns the
    spike trains

    PARAMETERS
    ----------
    network: .DynamicsAdaptation
        'abr_control.controllers.signals.dynamics_adaptation'
    input_signal: np array shape of (time_steps x input_dim)
        the input used for the network sim
    synapse: float, Optional (Default: None)
        the synapse filter on the nengo probe
    '''
    # if there aren't neuron probes in the network add them
    with network.nengo_model:
        network.probe_neurons = []
        for ens in network.adapt_ens:
            network.probe_neurons.append(
                nengo.Probe(ens.neurons, synapse=synapse))
    network.sim = nengo.Simulator(network.nengo_model, progress_bar=False)

    for in_sig in input_signal:
        network.input_signal = in_sig
        network.sim.run(dt, progress_bar=False)

    spike_trains = []
    for probe in network.probe_neurons:
        spike_trains.append(network.sim.data[probe] * dt)
    spike_trains = np.hstack(spike_trains)

    return np.array(spike_trains)


def proportion_neurons_responsive_to_input_signal(
        network, input_signal, ax=None):
    '''
    Accepts a Nengo network and checks the tuning curve responses to the input
    Plots the proportion of active neurons vs run time onto the ax object if
    provided

    Returns the proportion active and the activities

    PARAMETERS
    ----------
    network: .DynamicsAdaptation
        'abr_control.controllers.signals.dynamics_adaptation'
    input_signal: np array shape of (time_steps x input_dim)
        the input used for the network sim
    ax: ax object
        used for the rasterplot
    '''
    activities = get_activities(network=network, input_signal=input_signal)

    proportion_active = []
    for activity in activities:
        proportion_active.append(np.sum(activity, axis=1) /
                                 network.n_neurons)
    proportion_active = np.sum(
        proportion_active, axis=0) / len(proportion_active)

    if ax is not None:
        print('Plotting proportion of active neurons over time...')
        ax.plot(proportion_active, label='proportion active')

        ax.set_title('Proportion of active neurons over time')
        ax.set_ylabel('Proportion Active')
        ax.set_xlabel('Time steps')
        ax.set_ylim(0, 1)
        plt.legend()

    return proportion_active, activities


def proportion_neurons_active_over_time(
        input_signal=None, network=None, spike_trains=None, ax=None):
    '''
    Accepts a Nengo network and simulates its response to a given input
    Plots the proportion of active neurons vs run time onto the ax object
    Returns the proportion active and the spike trains

    PARAMETERS
    ----------
    input_signal: np array (time_steps x input_dim)
        the input used for the network sim
    network: .DynamicsAdaptation
        'abr_control.controllers.signals.dynamics_adaptation'
    spike_trains: np.array (timesteps x n_neurons), Optional (Default: None)
        the output from get_spike_trains()
    ax: ax object
        for plotting the output
    '''
    assert not (network is None and spike_trains is None), (
        "Either a network object or an array of spike trains must be provided")

    if spike_trains is None:
        spike_trains = get_spike_trains(
            network=network, input_signal=input_signal)

    n_neurons_active = np.zeros(spike_trains.shape[0])
    for ii, timestep in enumerate(spike_trains):
        n_neurons_active[ii] = len(np.where(timestep)[0])
    proportion_neurons_active = n_neurons_active / network.n_neurons

    if ax is not None:
        print('Plotting proportion of active neurons over time...')
        ax.plot(proportion_neurons_active, label='proportion active')

        ax.set_title('Proportion of active neurons over time')
        ax.set_ylabel('Proportion Active')
        ax.set_xlabel('Time steps')
        # ax.set_ylim(0, 1)
        plt.legend()

    return proportion_neurons_active, spike_trains


def proportion_time_neurons_active(
        input_signal=None, network=None, spike_trains=None, ax=None):
    '''
    Accepts a Nengo network andsimulates its response to a given input
    Plots a histogram of neuron activity relative to run time onto ax
    Returns the time active and the spike trains

    PARAMETERS
    ----------
    input_signal: np array (time_steps x input_dim)
        the input used for the network sim
    network: .DynamicsAdaptation, Optional, (Default: None)
        'abr_control.controllers.signals.dynamics_adaptation'
    spike_trains: np.array (timesteps x n_neurons), Optional (Default: None)
        the output from get_spike_trains()
    ax: ax object
        for plotting the output
    '''
    assert not (network is None and spike_trains is None), (
        "Either a network object or an array of spike trains must be provided")

    if spike_trains is None:
        spike_trains = get_spike_trains(
            network=network,
            input_signal=input_signal,
            synapse=network.tau_output)

    # for spike_train in spike_trains:
    n_timesteps_active = np.zeros(spike_trains.shape[1])
    for ii, timestep in enumerate(spike_trains.T):
        n_timesteps_active[ii] = len(np.where(timestep > 1e-5)[0])
    proportion_time_active = n_timesteps_active / spike_trains.shape[0]

    if ax is not None:
        plt.hist(proportion_time_active, bins=np.linspace(0, 1, 100))
        ax.set_ylabel('Number of active neurons')
        ax.set_xlabel('Proportion of Time')
        ax.set_title('Proportion of time neurons are active')

    return proportion_time_active, spike_trains


def n_neurons_active_and_inactive(activity):
    '''
    Accepts a list of activities set to 1's and 0's based on some
    thereshold in get_activities() and returns how many neurons are
    active and never active

    PARAMETERS
    ----------
    activity: int list of shape (n_timesteps x n_neurons)
        a list of activities represented as 1 for spiking and 0 for not
        spiking over all inputs passed to the nengo simulator, for each
        ensemble of the network
    '''
    activity = np.asarray(activity)
    if activity.ndim != 2:
        raise Exception('Input should be n_timesteps x n_neurons')

    activity_sum = np.sum(activity, axis=0)
    n_inactive = len(np.where(activity_sum == 0)[0])
    n_active = activity.shape[1] - n_inactive
    return n_active, n_inactive


def gen_learning_profile(network, input_signal, ax_list=None,
                         n_ens_to_raster=None, show_plot=True,
                         n_neurons_per_ens_to_plot=-1):
    """
    Plots the networks neural activity onto three subplots, the rasterplot,
    proportion of active neurons over time, and how many neurons were active
    over different proportions of run time

    Accepts an abr_control dynamics_adaptation network object and input signal
    Plots
    1. rasterplot showing spikes for each neuron over time on one axis, and the
       input signal of the other
    2. proportion of time active, the number of neurons active vs proportion
       of run time
    3. proportion of neurons that are active over time

    PARAMETERS
    ----------
    network: .DynamicsAdaptation
        'abr_control.controllers.signals.dynamics_adaptation'
    input_signal: np array shape of (time_steps x input_dim)
        the input used for the network sim
    ax_list: list of 3 ax objects
        used for the rasterplot
    n_ens_to_raster: int, Optional (Default: None)
        the number of ensembles to plot in the raster,
        if None all will be plotted
    show_plot: boolean, Optional (Default: True)
        whether to show the figure at the end of the script or not
    n_neurons_per_ens_to_plot: int, Optional (Default: -1)
        the number of neurons to plot per ensemble, -1 will plot all
    """

    if ax_list is None:
        plt.figure(figsize=(8, 15))
        ax_list = []
        for ii in range(0, 3):
            ax_list.append(plt.subplot(3, 1, ii+1))

    raster_plot(
        network=network,
        input_signal=input_signal,
        ax=ax_list[0],
        n_ens_to_raster=n_ens_to_raster,
        n_neurons_per_ens_to_plot=n_neurons_per_ens_to_plot)

    proportion_active, spike_trains = proportion_neurons_active_over_time(
        input_signal=input_signal,
        network=network,
        # spike_trains=spike_trains,
        ax=ax_list[1])

    proportion_time_neurons_active(
        # input_signal=input_signal,
        network=network,
        spike_trains=spike_trains,
        ax=ax_list[2])

    n_active, n_inactive = n_neurons_active_and_inactive(activity=spike_trains)

    print('Number of neurons inactive: ', n_inactive)
    print('Number of neurons active: ', n_active)
    ax_list[1].legend(['Mean Prop Active: %.2f'%np.mean(proportion_active)])
    ax_list[2].legend(['Active: %i  |  Inactive: %i'%(n_active, n_inactive)])

    if show_plot:
        plt.tight_layout()
        plt.show()


def gen_intercept_bounds_and_modes(intercept_range=None, intercept_step=0.1,
                                   mode_range=None, mode_step=0.2):
    '''
    Accepts a range of intercept bounds and modes and returns an np.array
    of the valid combinations

    The validity is based on the following rules:
    - left bound < right bound
    - mode >= left bound
    - mode <= right bound

    PARAMETERS
    ----------
    intercept_range: list of two floats, Optional (Default: [-0.9, 1])
        the range of bounds to try. *See Note at bottom*
    mode_range: list of two floats, Optional (Default: [-0.9, 1])
        the range of modes to try. *See Note at bottom*
    intercept_step: float, Optional (Default: 0.1)
        the step size for the range of values between the range specified
    mode_step: float, Optional (Default: 0.2)
        the step size for the range of values between the range specified

    NOTE:
        the way the range function used on these values works, the second
        value (far right of range) is ignored. For this reason, to include
        0.9, you must have the right side of the limit set to you desired
        limit + intercept_step.
        EX: to include 0.9 as the far right limit for the intercept bounds,
        assuming the intercept step is set to 0.1, the intercept range for
        the right bound must be set to 0.9 + 0.1 = 1.0 to check the range
        of values up to and including 0.9
    '''
    if intercept_range is None:
        intercept_range = [-0.9, 1]
    if mode_range is None:
        mode_range = [-0.9, 1]
    intercept_range = np.arange(intercept_range[0], intercept_range[1],
                                intercept_step)
    mode_range = np.arange(mode_range[0], mode_range[1], mode_step)

    # Create list of all possible intercepts
    intercepts = np.array(np.meshgrid(
        intercept_range, intercept_range)).T.reshape(-1, 2)
    # get a list of all valid intercepts
    valid = []
    for vals in intercepts:
        vals[0] = round(vals[0], 1)
        vals[1] = round(vals[1], 1)
        if vals[0] < vals[1]:
            for mode in mode_range:
                mode = round(mode, 1)
                if vals[0] <= mode <= vals[1]:
                    valid.append(np.array([vals[0], vals[1], mode]))

    intercepts = np.array(valid)
    print('There are %i valid combinations of intercepts and modes' %
          len(intercepts))

    return intercepts
