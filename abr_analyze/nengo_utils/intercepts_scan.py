'''
Accepts the parameters to instantiate a dynamics_adaptation network from
abr_control and a list of intercept ranges and modes. A simulation of each
possible intercept is run, passing in the input signal, to generate a neural
profile for each sim. The profiles can be viewed using the
intercept_scan_viewer.py gui
'''
import timeit
import matplotlib.pyplot as plt
import numpy as np

from abr_analyze.data_handler import DataHandler
import abr_analyze.nengo_utils.network_utils as network_utils
from abr_control.controllers import signals

def proportion_neurons_active(encoders, intercept_vals, input_signal, seed=1,
                              save_name='proportion_neurons', notes='',
                              **kwargs):
    '''
    runs a scan for the proportion of neurons that are active over time

    PARAMETERS
    ----------
    encoders: array of floats (n_neurons x n_inputs)
        the values that specify along what vector a neuron will be
        sensitive to
    intercept_vals: array of floats (n_intercepts to try x 3)
        the [left_bound, mode, right_bound] to pass on to the triangluar
        intercept function in network_utils
    input_signal: array of floats (n_timesteps x n_inputs)
        the input signal that we want to check our networks response to
    seed: int
        the seed used for any randomization in the sim
    save_name: string, Optional (Default: proportion_neurons)
        the name to save the data under in the intercept_scan database
    notes: string, Optional (Default: '')
        any additional notes to save with the scan
    '''
    dat = DataHandler('intercepts_scan')

    loop_time = 0
    for ii, intercept in enumerate(intercept_vals):

        start = timeit.default_timer()
        print('%.2f%% Complete, ~%.2f min remaining...' %
              (ii/len(intercept_vals)*100,
               (len(intercept_vals)-ii)*loop_time/60))

        # create our intercept distribution from the intercepts vals
        intercept_list = signals.AreaIntercepts(
            dimensions=encoders.shape[1],
            base=signals.Triangular(intercept[0], intercept[2], intercept[1]))
        rng = np.random.RandomState(seed)
        intercept_list = intercept_list.sample(encoders.shape[0], rng=rng)
        intercept_list = np.array(intercept_list)

        # create a network with the new intercepts
        network = signals.DynamicsAdaptation(
            n_input=encoders.shape[1],
            n_output=1,  # number of output is irrelevant
            n_neurons=encoders.shape[0],
            intercepts=intercept_list,
            probe_weights=True,
            seed=seed,
            encoders=encoders,
            **kwargs)

        # get the proportion of neurons active
        proportion_active, activity = (
            network_utils.prop_active_neurons_over_time(
                network=network, input_signal=input_signal))

        # get the number of active and inactive neurons
        num_active, num_inactive = (
            network_utils.num_neurons_active_and_inactive(
                activity=activity))

        # save the data for the line plot of the histogram
        x = np.cumsum(np.ones(len(proportion_active)))
        ideal = [ideal_function(x) for val in x]
        dat.save(
            data={'total_intercepts':len(intercept_vals), 'notes':notes},
            save_location='%s' % save_name,
            overwrite=True)

        # not saving activity because takes up a lot of disk space
        data = {'intercept_bounds': intercept[:2],
                'intercept_mode': intercept[2],
                'x': x,
                'y': proportion_active,
                'num_active': num_active,
                'num_inactive': num_inactive,
                'xlabel': 'Time steps',
                'ylabel': 'Proportion neurons active',
                }
        dat.save(data=data, save_location='%s/%05d'%(save_name, ii),
                 overwrite=True)
        loop_time = timeit.default_timer() - start

    # review(save_name=save_name, num_to_plot=10)


def proportion_time_active(encoders, intercept_vals, input_signal, seed=1,
                           save_name='proportion_time', n_bins=100, notes='',
                           **kwargs):
    '''
    runs a scan for to show how many neurons are active over different
    proportions of sim time

    PARAMETERS
    ----------
    seed: int
        the seed used for any randomization in the sim
    encoders: array of floats (n_neurons x n_inputs)
        the values that specify along what vector a neuron will be
        sensitive to
    input_signal: array of floats (n_timesteps x n_neurons)
        the input signal that we want to check our networks response to
    ideal_fuinction: lambda function(0 to 1)
        used as the desired profile to compare against. The review function
        will use this to find the closest matching results
    intercept_vals: array of floats (n_intercepts to try x 3)
        the [left_bound, mode, right_bound] to pass on to the triangluar
        intercept function in network_utils
    save_name: string, Optional (Default: proportion_neurons)
        the name to save the data under in the intercept_scan database
    n_bins: int, Optional (Default: 100)
        the number of time proportions to break up the neurons into
    notes: string, Optional (Default: '')
        any additional notes to save with the scan
    '''
    dat = DataHandler('intercepts_scan')

    loop_time = 0
    bins = np.linspace(0, 1, n_bins)
    for ii, intercept in enumerate(intercept_vals):

        start = timeit.default_timer()
        print('%.2f%% Complete, ~%.2f min remaining...' %
              ((ii/len(intercept_vals)*100),
               (len(intercept_vals)-ii)*loop_time/60))

        # create our intercept distribution from the intercepts vals
        intercept_list = signals.AreaIntercepts(
            dimensions=encoders.shape[1],
            base=signals.Triangular(intercept[0], intercept[2],
                                    intercept[1]))
        rng = np.random.RandomState(seed)
        intercept_list = intercept_list.sample(encoders.shape[0], rng=rng)
        intercept_list = np.array(intercept_list)

        # create a network with the new intercepts
        network = signals.DynamicsAdaptation(
            n_input=encoders.shape[1],
            n_output=1,  # number of output dimensions is irrelevant
            n_neurons=encoders.shape[0],
            intercepts=intercept_list,
            probe_weights=True,
            seed=seed,
            encoders=encoders,
            **kwargs)

        # get the time active
        time_active, activity = network_utils.prop_time_neurons_active(
            network=network, input_signal=input_signal)

        # get the number of active and inactive neurons
        num_active, num_inactive = (
            network_utils.num_neurons_active_and_inactive(
                activity=activity))

        # save the data for the line plot of the histogram
        y, bins_out = np.histogram(np.squeeze(time_active), bins=bins)
        centers = 0.5*(bins_out[1:]+bins_out[:-1])
        ideal = [ideal_function(x) for x in centers]
        dat.save(
            data={'ideal': ideal, 'total_intercepts': len(intercept_vals),
                  'notes': notes},
            save_location='%s' % save_name,
            overwrite=True)

        diff_to_ideal = ideal - y
        error = np.sum(np.abs(diff_to_ideal))

        # not saving activity because takes up a lot of disk space
        data = {'intercept_bounds':intercept[:2],
                'intercept_mode':intercept[2],
                'diff_to_ideal':diff_to_ideal,
                'x':centers,
                'y':y,
                'num_active':num_active,
                'num_inactive':num_inactive,
                'error':error,
                'xlabel':'proportion time active',
                'ylabel':'num neurons active'}
        dat.save(data=data, save_location='%s/%05d' % (save_name, ii),
                 overwrite=True)

        loop_time = timeit.default_timer() - start

    review(save_name=save_name, num_to_plot=10)


def review(save_name, ideal_function, num_to_plot=10):
    '''
    loads the data from save name and gets num_to_plot tests that most
    closley match the ideal function that was passed in during the scan

    PARAMETERS
    ----------
    save_name: string
        the location in the intercepts_scan database to load from
    ideal_fuinction: lambda function(n_timesteps)
        used as the desired profile to compare against. The review function
        will use this to find the closest matching results
    num_to_plot: int, Optional (Default: 10)
        the number of tests to find that most closley match the ideal
    '''
    dat = DataHandler('intercepts_scan')

    ideal_data = dat.load(parameters=['ideal', 'total_intercepts'],
                          save_location='%s' % save_name)
    ideal = ideal_data['ideal']
    num = ideal_data['total_intercepts']

    run_data = []
    errors = []
    for ii in range(0, num):
        data = dat.load(
            parameters=['intercept_bounds', 'intercept_mode',
                        'diff_to_ideal', 'x', 'y', 'error', 'num_active',
                        'num_inactive', 'xlabel', 'ylabel'],
            save_location='%s/%05d' % (save_name, ii))

        ideal = [ideal_function(x) for val in x]
        diff_to_ideal = ideal - y
        error = np.sum(np.abs(diff_to_ideal))

        run_data.append(data)
        errors.append(data['error'])

    indices = np.array(errors).argsort()[:num_to_plot]
    print('Plotting...')
    plt.figure()
    for ii in range(0, num_to_plot):
        ind = indices[ii]
        data = run_data[ind]
        #plt.title('Active: %i | Inactive: %i' % (data['num_active'],
        #          data['num_inactive']))
        plt.plot(np.squeeze(data['x']), np.squeeze(data['y']),
                 label='%i: err:%.2f \n%s: %s'%
                 (ind, data['error'], data['intercept_bounds'],
                  data['intercept_mode']))

    plt.xlabel(data['xlabel'])
    plt.ylabel(data['ylabel'])
    plt.plot(np.squeeze(data['x']), ideal, c='k', lw=3, linestyle='--',
             label='ideal')
    plt.legend()
    plt.show()
