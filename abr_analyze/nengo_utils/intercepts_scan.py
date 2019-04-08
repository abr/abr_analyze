'''
Accepts the parameters to instantiate a dynamics_adaptation network from
abr_control and a list of intercept ranges and modes. A simulation of each
possible intercept is run, passing in the input signal, to generate a neural
profile for each sim. The profiles can be viewed using the
intercept_scan_viewer.py gui
'''
import gc
import timeit
import matplotlib.pyplot as plt
import numpy as np

from abr_control.controllers import signals
from abr_analyze.data_handler import DataHandler
import abr_analyze.nengo_utils.network_utils as network_utils

@profile
def run(encoders, intercept_vals, input_signal, seed=1,
        save_name='example', notes='', analysis_fncs=None, **kwargs):
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
    if not isinstance(analysis_fncs, list):
        analysis_fncs = [analysis_fncs]

    loop_time = 0
    elapsed_time = 0
    for ii, intercept in enumerate(intercept_vals):
        start = timeit.default_timer()
        elapsed_time += loop_time
        print('%i/%i | '%(ii+1, len(intercept_vals))
              + '%.2f%% Complete | '%(ii/len(intercept_vals)*100)
              + '%.2f min elapsed | '%(elapsed_time/60)
              + '~%.2f min remaining...'
              %((len(intercept_vals)-ii)*loop_time/60),
              end='\r')

        # create our intercept distribution from the intercepts vals
        intercept_list = signals.AreaIntercepts(
            dimensions=encoders.shape[2],
            base=signals.Triangular(intercept[0], intercept[2], intercept[1]))
        rng = np.random.RandomState(seed)
        intercept_list = intercept_list.sample(encoders.shape[1], rng=rng)
        intercept_list = np.array(intercept_list)

        # create a network with the new intercepts
        network = signals.DynamicsAdaptation(
            n_input=encoders.shape[2],
            n_output=1,  # number of output is irrelevant
            n_neurons=encoders.shape[1],
            intercepts=intercept_list,
            probe_weights=True,
            seed=seed,
            encoders=encoders,
            **kwargs)

        # get the spike trains from the sim
        spike_trains = network_utils.get_spike_trains(
            network=network, input_signal=input_signal)

        # loop through the analysis functions
        for func in analysis_fncs:
            func_name = func.__name__
            y, activity = func(network=network, input_signal=input_signal,
                               spike_trains=spike_trains)

            # get the number of active and inactive neurons
            num_active, num_inactive = (
                network_utils.n_neurons_active_and_inactive(activity=activity))

            if ii == 0:
                dat = DataHandler('intercepts_scan')
                dat.save(
                    data={'total_intercepts':len(intercept_vals),
                          'notes':notes},
                    save_location='%s/%s'%(save_name, func_name),
                    overwrite=True)

            # not saving activity because takes up a lot of disk space
            data = {'intercept_bounds': intercept[:2],
                    'intercept_mode': intercept[2],
                    'y': y,
                    'num_active': num_active,
                    'num_inactive': num_inactive,
                    'title': func_name
                    }
            dat.save(data=data, save_location='%s/%s/%05d'%
                     (save_name, func_name, ii), overwrite=True)

            loop_time = timeit.default_timer() - start

        # clear memory
        start = None
        intercept_list = None
        network = None
        spike_trains = None
        y = None
        activity = None
        num_active = None
        num_inactive = None
        gc.collect()

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

    if num_to_plot > num:
        print('Only %i runs to plot' % num)
        num_to_plot = num

    run_data = []
    errors = []
    for ii in range(0, num):
        data = dat.load(
            parameters=['intercept_bounds', 'intercept_mode',
                        'y', 'error', 'num_active',
                        'num_inactive', 'title'],
            save_location='%s/%05d' % (save_name, ii))

        data['x'] = np.cumsum(np.ones(len(data['y'])))
        ideal = [ideal_function(x) for x in data['x']]
        diff_to_ideal = ideal - data['y']
        error = np.sum(np.abs(diff_to_ideal))

        run_data.append(data)
        errors.append(error)

    print('errors: ', errors)
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
                 (ind, error, data['intercept_bounds'],
                  data['intercept_mode']))

    plt.title(data['title'])
    plt.plot(np.squeeze(data['x']), ideal, c='k', lw=3, linestyle='--',
             label='ideal')
    plt.legend()
