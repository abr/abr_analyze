'''
Accepts the parameters to instantiate a dynamics_adaptation network from
abr_control and a list of intercept ranges and modes. A simulation of each
possible intercept is run, passing in the input signal, to generate a neural
profile for each sim. The profiles can be viewed using the
intercept_scan_viewer.py gui
'''
import matplotlib.pyplot as plt
import numpy as np
import timeit

from abr_analyze.data_handler import DataHandler
from abr_analyze.nengo_utils.network_utils import NetworkUtils
from abr_control.controllers import signals

class InterceptsScan():
    def __init__(self):
        self.net_utils = NetworkUtils()
        self.dat = DataHandler('intercepts_scan')

    def proportion_neurons_active(
            self, n_input, n_output, n_neurons, n_ensembles, pes_learning_rate,
            backend, seed, neuron_type, encoders, input_signal, ideal_function,
            intercept_vals, save_name='proportion_neurons', notes=''):
        '''
        runs a scan for the proportion of neurons that are active over time

        PARAMETERS
        ----------
        n_input: int
            the number of input dimensions
        n_output: int
            the number of output dimensions
        n_neurons: int
            the number of neurons in each ensemble
        n_ensembles: int
            the number of ensembles in the sim
        pes_learning_rate: float
            the learning rate for the simulation, note however that this
            simulation does not perform any learning so this value will not
            affect the results
        backend: string
            specifies what nengo backend to use, which are listed in the
            abr_control.controllers.signals.dynamics_adaptation() class
            'nengo_cpu', 'nengo_gpu', 'nengo_ocl'
        seed: int
            the seed used for any randomization in the sim
        neuron_type: string
            the type of neurons to use for the simulation
            'lif', 'relu'
        encoders: array of floats (n_ensembles x n_neurons)
            the values that specify along what vector a neuron will be
            sensitive to
        input_signal: array of floats (n_timesteps x n_neurons)
            the input signal that we want to check our networks response to
        ideal_fuinction: lambda function(n_timesteps)
            used as the desired profile to compare against. The review function
            will use this to find the closest matching results
        intercept_vals: array of floats (n_intercepts to try x 3)
            the [left_bound, mode, right_bound] to pass on to the triangluar
            intercept function in NetworkUtils
        save_name: string, Optional (Default: proportion_neurons)
            the name to save the data under in the intercept_scan database
        notes: string, Optional (Default: '')
            any additional notes to save with the scan
        '''

        loop_time = 0
        for ii ,intercept in enumerate(intercept_vals) :
            start = timeit.default_timer()
            print('%.2f%% Complete, ~%.2f min remaining...'%
                    ((ii/len(intercept_vals)*100),
                     (len(intercept_vals)-ii)*loop_time/60))#,
                     #end='\r')
            # create our intercept distribution from the intercepts vals
            intercept_list = signals.AreaIntercepts(
                dimensions=n_input,
                base=signals.Triangular(intercept[0], intercept[2], intercept[1]))
            rng = np.random.RandomState(seed)
            intercept_list = intercept_list.sample(n_neurons, rng=rng)
            intercept_list = np.array(intercept_list)

            # create a network with the new intercepts
            network = signals.DynamicsAdaptation(
                n_input=n_input,
                n_output=n_output,
                n_neurons=n_neurons,
                n_ensembles=n_ensembles,
                pes_learning_rate=1e-6,
                intercepts=intercept_list,
                backend=backend,
                probe_weights=True,
                seed=seed,
                neuron_type=neuron_type,
                encoders=encoders)

            # get the proportion of neurons active
            [proportion_active, activity] = self.net_utils.prop_active_neurons_over_time(
                                            network=network,
                                            input_signal=input_signal)

            # get the number of active and inactive neurons
            [num_active, num_inactive] = self.net_utils.num_neurons_active_and_inactive(
                                            activity=activity)

            # save the data for the line plot of the histogram
            x = np.cumsum(np.ones(len(proportion_active)))
            ideal = [ideal_function(x) for val in x]
            self.dat.save(
                    data={'ideal': ideal, 'total_intercepts':
                        len(intercept_vals), 'notes': notes},
                    save_location='%s'%save_name,
                    overwrite=True)

            diff_to_ideal = ideal - proportion_active
            error = np.sum(np.abs(diff_to_ideal))

            # not saving activity because takes up a lot of disk space
            data = {'intercept_bounds': intercept[:2],
                    'intercept_mode': intercept[2], 'diff_to_ideal': diff_to_ideal,
                    'x': x, 'y': proportion_active, 'num_active': num_active,
                    'num_inactive': num_inactive, 'error': error,
                    'xlabel': 'time steps', 'ylabel': 'proportion neurons active'}
            self.dat.save(data=data, save_location='%s/%05d'%(save_name, ii), overwrite=True)
            loop_time = timeit.default_timer() - start

        self.review(save_name=save_name, num_to_plot=10)


    def proportion_time_active(
            self, n_input, n_output, n_neurons, n_ensembles, pes_learning_rate,
            backend, seed, neuron_type, encoders, input_signal, ideal_function,
            intercept_vals, save_name='proportion_time', n_bins=100, notes=''):
        '''
        runs a scan for to show how many neurons are active over different
        proportions of sim time

        PARAMETERS
        ----------
        n_input: int
            the number of input dimensions
        n_output: int
            the number of output dimensions
        n_neurons: int
            the number of neurons in each ensemble
        n_ensembles: int
            the number of ensembles in the sim
        pes_learning_rate: float
            the learning rate for the simulation, note however that this
            simulation does not perform any learning so this value will not
            affect the results
        backend: string
            specifies what nengo backend to use, which are listed in the
            abr_control.controllers.signals.dynamics_adaptation() class
            'nengo_cpu', 'nengo_gpu', 'nengo_ocl'
        seed: int
            the seed used for any randomization in the sim
        neuron_type: string
            the type of neurons to use for the simulation
            'lif', 'relu'
        encoders: array of floats (n_ensembles x n_neurons)
            the values that specify along what vector a neuron will be
            sensitive to
        input_signal: array of floats (n_timesteps x n_neurons)
            the input signal that we want to check our networks response to
        ideal_fuinction: lambda function(0 to 1)
            used as the desired profile to compare against. The review function
            will use this to find the closest matching results
        intercept_vals: array of floats (n_intercepts to try x 3)
            the [left_bound, mode, right_bound] to pass on to the triangluar
            intercept function in NetworkUtils
        save_name: string, Optional (Default: proportion_neurons)
            the name to save the data under in the intercept_scan database
        n_bins: int, Optional (Default: 100)
            the number of time proportions to break up the neurons into
        notes: string, Optional (Default: '')
            any additional notes to save with the scan
        '''
        loop_time = 0
        bins = np.linspace(0,1,n_bins)
        for ii ,intercept in enumerate(intercept_vals) :
            start = timeit.default_timer()
            print('%.2f%% Complete, ~%.2f min remaining...'%
                    ((ii/len(intercept_vals)*100),
                     (len(intercept_vals)-ii)*loop_time/60))#,
                     #end='\r')
            # create our intercept distribution from the intercepts vals
            intercept_list = signals.AreaIntercepts(
                dimensions=n_input,
                base=signals.Triangular(intercept[0], intercept[2], intercept[1]))
            rng = np.random.RandomState(seed)
            intercept_list = intercept_list.sample(n_neurons, rng=rng)
            intercept_list = np.array(intercept_list)

            # create a network with the new intercepts
            network = signals.DynamicsAdaptation(
                n_input=n_input,
                n_output=n_output,
                n_neurons=n_neurons,
                n_ensembles=n_ensembles,
                pes_learning_rate=1e-6,
                intercepts=intercept_list,
                backend=backend,
                probe_weights=True,
                seed=seed,
                neuron_type=neuron_type,
                encoders=encoders)

            # get the time active
            [time_active, activity] = self.net_utils.prop_time_neurons_active(
                                            network=network,
                                            input_signal=input_signal)

            # get the number of active and inactive neurons
            [num_active, num_inactive] = self.net_utils.num_neurons_active_and_inactive(
                                            activity=activity)

            # save the data for the line plot of the histogram
            y, bins_out = np.histogram(np.squeeze(time_active), bins=bins)
            centers = 0.5*(bins_out[1:]+bins_out[:-1])
            ideal = [ideal_function(x) for x in centers]
            self.dat.save(
                    data={'ideal': ideal, 'total_intercepts':
                        len(intercept_vals), 'notes': notes},
                    save_location='%s'%save_name,
                    overwrite=True)

            diff_to_ideal = ideal - y
            error = np.sum(np.abs(diff_to_ideal))

            # not saving activity because takes up a lot of disk space
            data = {'intercept_bounds': intercept[:2],
                    'intercept_mode': intercept[2], 'diff_to_ideal': diff_to_ideal,
                    'x': centers, 'y': y, 'num_active': num_active,
                    'num_inactive': num_inactive, 'error': error,
                    'xlabel': 'proportion time active', 'ylabel': 'num neurons active'}
            self.dat.save(data=data, save_location='%s/%05d'%(save_name, ii), overwrite=True)

            loop_time = timeit.default_timer() - start

        self.review(save_name=save_name, num_to_plot=10)

    def review(self, save_name, num_to_plot=10):
        '''
        loads the data from save name and gets num_to_plot tests that most
        closley match the ideal function that was passed in during the scan

        PARAMETERS
        ----------
        save_name: string
            the location in the intercepts_scan database to load from
        num_to_plot: int, Optional (Default: 10)
            the number of tests to find that most closley match the ideal
        '''
        ideal_data = self.dat.load(parameters=['ideal', 'total_intercepts'],
                save_location='%s'%save_name)
        ideal = ideal_data['ideal']
        num = ideal_data['total_intercepts']

        run_data = []
        errors = []
        for ii in range(0, num) :
            data = self.dat.load(
                    parameters=['intercept_bounds', 'intercept_mode', 'diff_to_ideal',
                    'x', 'y', 'error', 'num_active', 'num_inactive', 'xlabel',
                    'ylabel'],
                    save_location='%s/%05d'%(save_name,ii))
            run_data.append(data)
            errors.append(data['error'])

        indices = np.array(errors).argsort()[:num_to_plot]
        print('Plotting...')
        plt.figure()
        for ii in range(0, num_to_plot):
            ind = indices[ii]
            data = run_data[ind]
            #plt.title('Active: %i | Inactive: %i'%(data['num_active'], data['num_inactive']))
            plt.plot(np.squeeze(data['x']), np.squeeze(data['y']),
                    label='%i: err:%.2f \n%s: %s'%
                    (ind, data['error'], data['intercept_bounds'], data['intercept_mode']))

        plt.xlabel(data['xlabel'])
        plt.ylabel(data['ylabel'])
        plt.plot(np.squeeze(data['x']), ideal, c='k', lw=3, linestyle='--', label='ideal')
        plt.legend()
        plt.show()
