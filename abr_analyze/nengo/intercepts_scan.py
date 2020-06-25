'''
Accepts the parameters to instantiate a dynamics_adaptation network from
abr_control and a list of intercept ranges and modes. A simulation of each
possible intercept is run, passing in the input signal, to generate a neural
profile for each sim. The profiles can be viewed using the
intercept_scan_viewer.py gui
'''
import nengo
import timeit
import matplotlib.pyplot as plt
import numpy as np

from abr_control.controllers import signals
from nengo_extras import dists
from abr_analyze.data_handler import DataHandler
import abr_analyze.nengo.network_utils as network_utils

import nengo


def run(encoders, intercept_vals, input_signal, seed=1,
        db_name='intercepts_scan', save_name='example', notes='',
        analysis_fncs=None, network_class=None, network_ens_type=None,
        force_params=None, angle_params=None, means=None, variances=None, **kwargs):
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
    analysis_fncs: list of network_utils functions to apply to the spike trains
        the function must accept network and input signal, and return a list of
        data and activity
    '''
    if network_class is None:
        network_class = signals.DynamicsAdaptation
    if not isinstance(analysis_fncs, list):
        analysis_fncs = [analysis_fncs]

    print('Running intercepts scan on %s' % network_class.__name__)
    print('Input Signal Shape: ', np.asarray(input_signal).shape)

    loop_time = 0
    elapsed_time = 0
    for ii, intercept in enumerate(intercept_vals):
        start = timeit.default_timer()
        elapsed_time += loop_time
        print('%i/%i | ' % (ii+1, len(intercept_vals))
              + '%.2f%% Complete | ' % (ii/len(intercept_vals)*100)
              + '%.2f min elapsed | ' % (elapsed_time/60)
              + '%.2f min for last sim | ' % (loop_time/60)
              + '~%.2f min remaining...'
              % ((len(intercept_vals)-ii)*loop_time/60),
              end='\r')

        # create our intercept distribution from the intercepts vals
        # Generates intercepts for a d-dimensional ensemble, such that, given a
        # random uniform input (from the interior of the d-dimensional ball), the
        # probability of a neuron firing has the probability density function given
        # by rng.triangular(left, mode, right, size=n)
        np.random.seed(seed)
        triangular = np.random.triangular(
            # intercept_vals = [left, right, mode]
            left=intercept[0],
            right=intercept[1],
            mode=intercept[2],
            size=encoders.shape[1],
        )
        intercepts = nengo.dists.CosineSimilarity(encoders.shape[2] + 2).ppf(1 - triangular)
        intercept_list = intercepts.reshape((1, encoders.shape[1]))

        print()
        print(intercept)
        print(intercept_list)

        # create a network with the new intercepts
        # network = network_class(
            # n_input=encoders.shape[2],
            # n_output=1,  # number of output is irrelevant
            # n_neurons=encoders.shape[1],
            # intercepts=intercept_list,
            # seed=seed)#,
            # encoders=encoders,
            # **kwargs)

        # get the spike trains from the sim
        network = network_class(
            force_params=force_params,
            angle_params=angle_params,
            means=means,
            variances=variances,
            seed=seed)

        if network_ens_type == 'force':
           network_ens = network.force_ens
           synapse = force_params['tau_output']
        elif network_ens_type == 'angle':
            network_ens = network.angle_ens
            synapse = angle_params['tau_output']

        spike_trains = network_utils.get_activities(
            network=network, network_ens=network_ens,
            input_signal=input_signal,
            synapse=synapse)

        for func in analysis_fncs:
            func_name = func.__name__
            y, activity = func(
                    pscs=spike_trains,
                    n_neurons=n_neurons,
                    n_ensembles=n_ensembles)

            # get the number of active and inactive neurons
            num_active, num_inactive = (
                network_utils.n_neurons_active_and_inactive(activity=activity))

            if ii == 0:
                dat = DataHandler(db_name)
                dat.save(
                    data={'total_intercepts': len(intercept_vals),
                          'notes': notes},
                    save_location='%s/%s' % (save_name, func_name),
                    overwrite=True)

            # not saving activity because takes up a lot of disk space
            data = {'intercept_bounds': intercept[:2],
                    'intercept_mode': intercept[2],
                    'y': y,
                    'num_active': num_active,
                    'num_inactive': num_inactive,
                    'title': func_name
                    }
            dat.save(data=data, save_location='%s/%s/%05d' %
                     (save_name, func_name, ii), overwrite=True)

            loop_time = timeit.default_timer() - start


def review(save_name, ideal_function, num_to_plot=10, db_name='intercepts_scan'):
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
    dat = DataHandler(db_name)

    ideal_data = dat.load(parameters=['ideal', 'total_intercepts'],
                          save_location='%s' % save_name)
    ideal = ideal_data['ideal']
    num = ideal_data['total_intercepts']

    if num_to_plot > num:
        print('Only %i runs to plot' % num)
        num_to_plot = num

    run_data = []
    errors = []
    n_bins = 30
    for ii in range(0, num):
        data = dat.load(
            parameters=['intercept_bounds', 'intercept_mode',
                        'y', 'error', 'num_active',
                        'num_inactive', 'title'],
            save_location='%s/%05d' % (save_name, ii))

        if data['title'] == 'proportion_time_neurons_active':
            y, bins_out = np.histogram(np.squeeze(data['y']),
                                       bins=np.linspace(0, 1, n_bins))
            data['x'] = 0.5*(bins_out[1:]+bins_out[:-1])
            data['y'] = y
        else:
            data['x'] = np.cumsum(np.ones(len(data['y'])))

        ideal = [ideal_function(x) for x in data['x']]
        diff_to_ideal = ideal - data['y']
        error = np.sum(np.abs(diff_to_ideal))

        run_data.append(data)
        errors.append(error)

    indices = np.array(errors).argsort()[:num_to_plot]
    print('Plotting...')
    plt.figure()
    for ii in range(0, num_to_plot):
        ind = indices[ii]
        data = run_data[ind]
        if data['title'] == 'proportion_time_neurons_active':
            plt.bar(data['x'], data['y'], width=1/(2*n_bins),
                    edgecolor='white', alpha=0.5,
                    label=('%i: err:%.2f \n%s: %s' %
                           (ind, errors[ind], data['intercept_bounds'],
                            data['intercept_mode'])))
        else:
            plt.plot(np.squeeze(data['x']), np.squeeze(data['y']),
                     label=('%i: err:%.2f \n%s: %s' %
                            (ind, errors[ind], data['intercept_bounds'],
                             data['intercept_mode'])))

    plt.title(data['title'])
    plt.plot(np.squeeze(data['x']), ideal, c='k', lw=3, linestyle='--',
             label='ideal')
    plt.legend()
