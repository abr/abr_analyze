"""
using the input from a provided test and the desired upper and lower limits of
a triangular distribution, the script will run through every possible set of
intercepts and modes. The sets that provide activity that matches the ideal the
closest will be plotted along with their intercept values.
"""
from abr_analyze.utils import DataHandler, NetworkUtils
from abr_control.controllers import signals
import numpy as np
import time
import timeit
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class InterceptsScan():
    def __init__(self):
        self.net_utils = NetworkUtils()
        self.dat = DataHandler('intercepts_scan')

    def proportion_neurons_active(
            self, n_input, n_output, n_neurons, n_ensembles, pes_learning_rate,
            backend, seed, neuron_type, encoders, input_signal, ideal_function,
            intercept_vals, save_name='proportion_neurons', notes=''):

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
        # Plot the activity for the 5 sets of intercepts with the least deviation from
        # the ideal
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
