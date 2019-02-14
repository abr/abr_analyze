import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import os

import nengo
import nengolib
from nengo.utils.matplotlib import rasterplot

class NetworkUtils:
    def convert_to_spherical(self, input_signal):
        """
        accepts data of shape time x N_dim and returns the values converted
        into spherical space"""
        x = input_signal.T
        pi = np.pi
        spherical = []

        def scale(input_signal):
            """
            Takes inputs in the range of -1 to 1 and scales them to the range of
            0-pi, except for the last dimension which gets scaled to 0-2pi.
            This is the expected range of inputs prior to the conversion to
            spherical
            """
            signal = np.copy(input_signal)
            factor = pi
            for ii, dim in enumerate(input_signal):
                if ii == len(input_signal)-1:
                    factor = 2*pi
                signal[ii] = dim * factor# + factor
            return signal

        def sin_product(input_signal, count):
            """
            Handles the sin terms in the conversion to spherical coordinates where
            we multiple by sin(x_i) n-1 times
            """
            tmp = 1
            for jj in range(0, count):
                tmp *= np.sin(input_signal[jj])
            return tmp

        # nth input scaled to 0-2pi range, remainder from 0-pi
        # cycle through each input
        x_rad = scale(input_signal=x)

        for ss in range(0, len(x)):
            sphr = sin_product(input_signal=x_rad, count=ss)
            sphr*= np.cos(x_rad[ss])
            spherical.append(sphr)
        spherical.append(sin_product(input_signal=x_rad, count=len(x)))
        spherical = np.array(spherical).T
        return(spherical)

    def generate_encoders(self, input_signal=None, n_neurons=1000, thresh=0.008):
        """
        Accepts inputs signal in the shape of time X dim and outputs encoders
        for the specified number of neurons by sampling from the input.
        The selection is made by choosing inputs randomly and checking that
        they are minimally thresh away from one another to avoid overly similar
        encoders. If we have exhausted the search then we increase thresh and
        rescan, until len(encoders) == number of neurons. If there are not
        enough input signal samples to select for the number of neurons, the
        remainder will be filled with a selection from a scattered hypersphere
        """
        #TODO: scale thresh based on dimensionality of input, 0.008 for 2DOF
        # and 10k neurons, 10DOF 10k use 0.08, for 10DOF 100 went up to 0.708 by end
        # 0.3 works well for 1000
        # first run so we need to generate encoders for the sessions
        print('input_signal_length: ', len(input_signal))
        ii = 0
        same_count = 0
        prev_index = 0
        while (input_signal.shape[0] > n_neurons):
            if ii%1000 == 0:
                print(input_signal.shape)
                print('thresh: ', thresh)
            # choose a random set of indices
            n_indices = input_signal.shape[0]
            # make sure we're dealing with an even number
            n_indices -= 0 if ((n_indices % 2) == 0) else 1
            n_half = int(n_indices / 2)

            randomized_indices = np.random.permutation(range(n_indices))
            a = randomized_indices[:n_half]
            b = randomized_indices[n_half:]

            data1 = input_signal[a]
            data2 = input_signal[b]

            distances = np.linalg.norm(data1 - data2, axis=1)

            under_thresh = distances > thresh

            input_signal = np.vstack([data1, data2[under_thresh]])
            ii += 1
            if prev_index == n_indices:
                same_count += 1
            else:
                same_count = 0

            if same_count == 50:
                same_count = 0
                thresh += 0.001
                print('All values are within threshold, but not at target size.')
                print('Increasing threshold to %.4f' %thresh)
            prev_index = n_indices

        if input_signal.shape[0] != n_neurons:
            print('Too many indices removed, appending with uniform hypersphere')
            print('shape: ', input_signal.shape)
            length = n_neurons - input_signal.shape[0]
            hypersphere = nengolib.stats.ScatteredHypersphere(surface=True)
            hyper_inputs = hypersphere.sample(length, input_signal.shape[1])
            input_signal = np.vstack((input_signal, hyper_inputs))


        print(input_signal.shape)
        print('thresh: ', thresh)
        encoders = np.array(input_signal)
        print(encoders.shape)
        return encoders

    def generate_scaled_inputs(self, q, dq, in_index):
        '''
        pass q dq in as time x dim shape
        accepts the 6 joint positions and velocities of the jaco2 and does the
        mean subtraction and scaling. Can set which joints are of interest with
        in_index, if it is not passed in the self.in_index instantiated in
        __init_network__ will be used

        returns two n x 6 lists scaled, one for q and one for dq

        '''
        # check if we received a 1D input (one timestep) or a 2D input (list of
        # inputs over time)
        # if np.squeeze(q)[0] > 1 and np.squeeze(q)[1] > 1:
        #     print('Scaling list of inputs')
        qs = q.T
        dqs = dq.T
        #print('raw q: ', np.array(qs).T.shape)

        # add bias to joints 0 and 4 so that the input signal doesn't keep
        # bouncing back and forth over the 0 to 2*pi line
        qs[0] = (qs[0] + np.pi) % (2*np.pi)
        qs[4] = (qs[4] + np.pi) % (2*np.pi)

        MEANS = {  # expected mean of joint angles / velocities
            # shift from 0-2pi to -pi to pi
            'q': np.array([3.20, 2.14, 1.52, 4.68, 3.00, 3.00]),
            'dq': np.array([0.002, -0.117, -0.200, 0.002, -0.021, 0.002]),
            }
        SCALES = {  # expected variance of joint angles / velocities
            'q': np.array([0.2, 1.14, 1.06, 1.0, 2.8, 0.01]),
            'dq': np.array([0.06, 0.45, 0.7, 0.25, 0.4, 0.01]),
            }

        for pp in range(0, 6):
            qs[pp] = (qs[pp] - MEANS['q'][pp]) / SCALES['q'][pp]
            dqs[pp] = (dqs[pp] - MEANS['dq'][pp]) / SCALES['dq'][pp]

        qs = qs
        dqs = dqs
        scaled_q = []
        scaled_dq = []
        #print(in_index)
        for ii in in_index:
            scaled_q.append(qs[ii])
            scaled_dq.append(dqs[ii])
        scaled_q = np.array(scaled_q).T
        scaled_dq = np.array(scaled_dq).T
        print('scaled q: ', np.array(scaled_q).shape)
        print('scaled dq: ', np.array(scaled_dq).shape)

        return [scaled_q, scaled_dq]

    def raster_plot(self, network, input_signal, ax, num_ens_to_raster=None):
        '''
        Accepts a Nengo network and runs a simulation with the provided input_signal
        Plots rasterplot onto ax object up to num_ens_to_raster ensembles
        if num_ens_to_raster is None, all ensembles will be plotted

        PARAMETERS
        ----------
        network: a Nengo network object
        input_signal: [time x input_dim] list
            the input used for the network sim
        ax: ax object
            used for the rasterplot
        num_ens_to_raster: int, Optional (Default: None)
            the number of ensembles to plot in the raster, if None all will be plotted
        '''
        # create probes to get rasterplot
        with network.nengo_model:
            if not hasattr(network.nengo_model, 'ens_probes'):
                network.nengo_model.ens_probes = []
                for ens in network.adapt_ens:
                    network.nengo_model.ens_probes.append(nengo.Probe(ens.neurons,
                            synapse=None))

        sim = nengo.Simulator(network.nengo_model)
        print('Running sim...')
        for ii, inputs in enumerate(input_signal):
            network.input_signal = inputs
            sim.run(time_in_seconds=0.001, progress_bar=False)
        print('Sim complete')

        probes = []
        print('Plotting spiking activity...')
        for ii, probe in enumerate(network.nengo_model.ens_probes):
            probes.append(sim.data[probe])
            # it can become hard to read the plot if there are too many ensembles
            # plotted onto one raster plot, let the user choose how many to plot
            if num_ens_to_raster is not None:
                if num_ens_to_raster == ii+1:
                    break

        probes = np.hstack(probes)
        time = np.ones(len(input_signal))
        ax = rasterplot(np.cumsum(time),probes, ax=ax)
        ax.set_ylabel('Neuron')
        ax.set_xlabel('Time [sec]')
        ax.set_title('Spiking Activity')

    def prop_active_neurons_over_time(self, network, input_signal, ax=None, thresh=None):
        '''
        Accepts a Nengo network and checks the tuning curve responses to the input signal
        Plots the proportion of active neurons vs run time onto the ax object if provided
        Returns the proportion active and the activities if ax is None

        PARAMETERS
        ----------
        network: a Nengo network object
        input_signal: [time x input_dim] list
            the input used for the network sim
        ax: ax object, Optional (Default: None)
            if None then the prop active and activities will be returned
            if provided will plot onto ax
        thresh: float, Optional (Default: None)
            the values above and below which activities get set to 1 and 0, respectively
            When None, the default of the function will be used
        '''
        time = np.ones(len(input_signal))
        activities = self.get_activities(network=network, input_signal=input_signal)

        proportion_active = []
        for activity in activities:
            # axis=0 mean over time
            # axis=1 mean over neurons
            # len(activity.T) gives the number of neurons
            proportion_active.append(np.sum(activity, axis=1)/len(activity.T))
        proportion_active = np.sum(proportion_active,
                axis=0)/len(proportion_active)

        if ax is not None:
            print('Plotting proportion of active neurons over time...')
            ax.plot(np.cumsum(time), proportion_active, label='proportion active')

            ax.set_title('Proportion of active neurons over time')
            ax.set_ylabel('Proportion Active')
            ax.set_xlabel('Time steps')
            ax.set_ylim(0, 1)
            plt.legend()

        return(proportion_active, activities)

    def prop_time_neurons_active(self, network, input_signal, ax=None, thresh=None):
        '''
        Accepts a Nengo network and checks the tuning curves response to the input signal
        Plots the the number of active neurons vs proportion of run time onto the ax object
        if provided, otherwise Returns the time active and the activities

        PARAMETERS
        ----------
        network: a Nengo network object
        input_signal: [time x input_dim] list
            the input used for the network sim
        ax: ax object, Optional (Default: None)
            if None then the prop active and activities will be returned
            if provided will plot onto ax
        thresh: float, Optional (Default: None)
            the values above and below which activities get set to 1 and 0, respectively
            When None, the default of the function will be used
        '''
        time = np.ones(len(input_signal))
        activities = self.get_activities(network=network, input_signal=input_signal)

        time_active = []
        for activity in activities:
            # axis=0 mean over time
            # axis=1 mean over neurons
            # len(activity.T) gives the number of neurons
            time_active.append(np.sum(activity, axis=0)/len(activity))
        time_active = np.hstack(time_active)

        if ax is not None:
            plt.hist(time_active, bins=np.linspace(0,1,100))
            ax.set_ylabel('Number of active neurons')
            ax.set_xlabel('Proportion of Time')
            ax.set_title('Proportion of time neurons are active')

        return (time_active, activities)

    def get_activities(self, network, input_signal, thresh=1e-5):
        '''
        Accepts a Nengo network and input signal and returns a list of the neural
        activities set to 1 or 0 based on the set thresh

        PARAMETERS
        ----------
        network: a Nengo network object
        input_signal: [time x input_dim] list
            the input used for the network sim
        thresh: float, Optional (Default: 1e-5)
            the values above and below which activities get set to 1 and 0, respectively
        '''

        activities = []
        for ens in network.adapt_ens:
            _, activity = nengo.utils.ensemble.tuning_curves(ens,
                    network.sim, input_signal)
            activity[activity>thresh]=1
            activity[activity<=thresh]=0
            activities.append(np.copy(activity))
        return activities

    def num_neurons_active_and_inactive(self, activity):
        # check how many neurons are never active
        num_inactive = 0
        num_active = 0
        for ens in activity:
            ens = ens.T
            for nn, neuron in enumerate(ens):
                if np.sum(ens[nn]) == 0:
                    num_inactive += 1
                else:
                    num_active += 1
        return [num_active, num_inactive]



    def gen_learning_profile(self, network, input_signal, ax=None, num_ens_to_raster=None,
            thresh=None, show_plot=True):
        """
        Plots the networks neural activity onto three subplots, showing the rasterplot,
        proportion of active neurons over time, and how many neurons were active over
        different proportions of run time

        Accepts a Nengo network and input signal
        Plots
        1. rasterplot showing spikes for each neuron over time on one axis, and the
           input signal of the other
        2. proportion of time active, the number of neurons active vs proportion
           of run time
        3. proportion of neurons that are active over time

        PARAMETERS
        ----------
        network: a Nengo network object
        input_signal: [time x input_dim] list
            the input used for the network sim
        ax: list of 3 ax objects, Optional (Default: None)
            if the three ax objects are not provided, they will be created
        num_ens_to_raster: int, Optional (Default: None)
            the number of ensembles to plot in the raster, if None all will be plotted
        thresh: float, Optional (Default: None)
            the values above and below which activities get set to 1 and 0, respectively
            When None, the default of the function will be used
        show_plot: boolean, Optional (Default: True)
            whether to show the figure at the end of the script or not
        """

        if ax is None:
            plt.figure(figsize=(8,15))
            ax = []
            for ii in range(0,3):
                ax.append(plt.subplot(3,1,ii+1))

        self.raster_plot(
                network=network,
                input_signal=input_signal,
                ax=ax[0],
                num_ens_to_raster=num_ens_to_raster)

        self.prop_active_neurons_over_time(
                network=network,
                input_signal=input_signal,
                ax=ax[1],
                thresh=thresh)

        [__, activity] = self.prop_time_neurons_active(
                network=network,
                input_signal=input_signal,
                ax=ax[2],
                thresh=thresh)

        [num_active, num_inactive] = self.num_neurons_active_and_inactive(
                                        activity=activity)
        print('Number of neurons inactive: ', num_inactive)
        print('Number of neurons active: ', num_active)
        ax[2].set_title('Proportion of time neurons are active\n'
                + 'Active: %i  |  Inactive: %i'%(num_active, num_inactive))

        if show_plot:
            plt.tight_layout()
            plt.show()

    def gen_intercept_bounds_and_modes(
            self, intercept_range=[-0.9,1], intercept_step=0.1,
            mode_range=[-0.9, 1], mode_step=0.2):

        intercept_range=np.arange(intercept_range[0], intercept_range[1],
                intercept_step)
        mode_range=np.arange(mode_range[0], mode_range[1], mode_step)

        # Create list of all possible intercepts
        intercepts = np.array(np.meshgrid(intercept_range, intercept_range)).T.reshape(-1, 2)
        # get a list of all valid intercepts
        valid = []
        rej = []
        for vals in intercepts:
            vals[0] = round(vals[0], 1)
            vals[1] = round(vals[1], 1)
            if vals[0] < vals[1]:
                for mode in mode_range:
                    mode = round(mode, 1)
                    if vals[0] <= mode and mode <= vals[1]:
                        valid.append(np.array([vals[0],vals[1], mode]))

        intercepts = np.array(valid)
        print('There are %i valid combinations of intercepts and modes'%len(intercepts))
        return intercepts


