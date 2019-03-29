'''
- a test function is made for every function in the data_processor
- various permutations and cases are tested for each function
- a dictionary is created with entries for every test
  - each test has it's own subtests for the various cases and permutations
  - the subtests are saved with the value of the boolean 'passed'
    which is true if the test passed or failed as expected
  - this is done by placing each subtest in a try except statement.
    this tests if an exception is raised, setting 'passed' based on the
    expected behaviour.
  - further tests are placed after the try except statement,
    specified for each function (ex: testing if a renamed group exists).
'''
import matplotlib.pyplot as plt
import numpy as np
import pytest

from abr_analyze.data_handler import DataHandler
import abr_analyze.data_processor as proc


# def test_get_mean_and_ci():
#     results = {}
#     test = 'test_get_mean_and_ci()'
#     results[test] = {}


@pytest.mark.parametrize('functions', (
    [np.sin], [np.sin, np.cos],
    )
)
def test_list_to_function(functions, plt):
    samples = 44
    x = np.linspace(0, 6.28, 100)
    x_step = x[-1]/100 * np.ones(100)
    ys = np.array([func(x) for func in functions]).T

    funcs = proc.list_to_function(data=ys, time_intervals=x_step)

    x_new = np.linspace(x_step[0], x[-1], samples)
    for y, func in zip(ys.T, funcs):
        plt.plot(x, y, 'o')
        plt.plot(x_new, func(x_new))
    plt.legend()
    plt.tight_layout()
    plt.show()


# def test_interpolate_data(data, time_intervals, interpolated_samples, axis):
#     dat = DataHandler('tests')
#     proc = DataProcessor()
#
#     y2 = proc.interpolate_data(
#         data=data,
#         time_intervals=time_intervals,
#         interpolated_samples=interpolated_samples,
#         axis=axis)
#         except Exception as e:
#             print('TEST: %s | SUBTEST: %s'%(test, label))
#             print('%s%s%s'%(RED,e,ENDC))
#             passed = not default_pass
#         results[test]['%s'%label] = passed
#         return results, y2
#
#     fig = plt.figure()
#     ax = []
#     for ii in range(0,4):
#         ax.append(fig.add_subplot(2,2,ii+1))
#
#     # passing in 1D list
#     y = [1, 2, 3.2, 5.5, 12, 20, 32]
#     x = np.ones(len(y))
#     samples = 30
#     results, y2 = interpolate_data(
#                     data=y,
#                     time_intervals=x,
#                     interpolated_samples=samples,
#                     axis=0,
#                     default_pass=True,
#                     results=results,
#                     label='1D list')
#     ax[0].plot(np.cumsum(x), y, 'o', label='raw')
#     ax[0].plot(np.linspace(x[0], np.sum(x), samples), y2)
#     ax[0].set_title('Data as 1D list')
#     ax[0].legend()
#     # passing in 2D list
#     y = ([[1, 2, 3.2, 5.5, 12, 20, 32],
#           [44, 54, 63, 77, 92, 111, 140]])
#     x = np.ones(len(y[0]))
#     results, y2 = interpolate_data(
#                     data=y,
#                     time_intervals=x,
#                     interpolated_samples=samples,
#                     axis=1,
#                     default_pass=True,
#                     results=results,
#                     label='2D list')
#     ax[1].plot(np.cumsum(x), y[0], 'o', label='raw')
#     ax[1].plot(np.cumsum(x), y[1], 'o', label='raw')
#     ax[1].plot(np.linspace(x[0], np.sum(x), samples), y2[ 0])
#     ax[1].plot(np.linspace(x[0], np.sum(x), samples), y2[ 1])
#     ax[1].set_title('Data as 2D list')
#     ax[1].legend()
#     # passing in 1D array
#     y = np.array([1, 2, 3.2, 5.5, 12, 20, 32])
#     x = np.ones(len(y))
#     results, y2 = interpolate_data(
#                     data=y,
#                     time_intervals=x,
#                     interpolated_samples=samples,
#                     axis=0,
#                     default_pass=True,
#                     results=results,
#                     label='1D array')
#     ax[2].plot(np.cumsum(x), y, 'o', label='raw')
#     ax[2].plot(np.linspace(x[0], np.sum(x), samples), y2)
#     ax[2].set_title('Data as 1D array')
#     ax[2].legend()
#     # passing in 2D array
#     y = np.array([[1, 2, 3.2, 5.5, 12, 20, 32],
#           [44, 54, 63, 77, 92, 111, 140]])
#     x = np.ones(len(y[0]))
#     results, y2 = interpolate_data(
#                     data=y,
#                     time_intervals=x,
#                     interpolated_samples=samples,
#                     axis=1,
#                     default_pass=True,
#                     results=results,
#                     label='2D array')
#     ax[3].plot(np.cumsum(x), y[0], 'o', label='raw')
#     ax[3].plot(np.cumsum(x), y[1], 'o', label='raw')
#     ax[3].plot(np.linspace(x[0], np.sum(x), samples), y2[0])
#     ax[3].plot(np.linspace(x[0], np.sum(x), samples), y2[1])
#     ax[3].set_title('Data as 2D array')
#     ax[3].legend()
#     plt.tight_layout()

#
# def test_scale_data():
#     results = {}
#     test = 'test_scale_data()'
#     results[test] = {}
#     print('\n%s----------%s----------%s'%(BLUE, test, ENDC))
#
#     def scale_data(data, baseline_low, baseline_high, scaling_factor,
#             test, label, default_pass, results):
#         try:
#             passed = default_pass
#             scaled = proc.scale_data(
#                 data=data,
#                 baseline_low=baseline_low,
#                 baseline_high=baseline_high,
#                 scaling_factor=scaling_factor)
#
#         except Exception as e:
#             print('TEST: %s | SUBTEST: %s'%(test, label))
#             print('%s%s%s'%(RED,e,ENDC))
#             passed = not default_pass
#             func = None
#         results[test]['%s'%label] = passed
#         return results, scaled
#
#     fig = plt.figure()
#     ax = []
#     for ii in range(0,2):
#         ax.append(fig.add_subplot(2, 1, ii+1))
#
#     # passing in 1D list
#     y_low = np.random.uniform(2, 4, 100)
#     y_high = np.random.uniform(7, 9, 100)
#     y = np.random.uniform(4, 7, 100)
#     scale = 1
#     results, y_scaled = scale_data(
#         data=y,
#         baseline_low=y_low,
#         baseline_high=y_high,
#         scaling_factor=scale,
#         test=test,
#         default_pass=True,
#         results=results,
#         label='Scale wrt to baselines')
#
#     ax[0].set_title('Raw data')
#     ax[0].plot(y, label='data')
#     ax[0].plot(y_low, label='low baseline')
#     ax[0].plot(y_high, label='high baseline')
#     ax[0].legend()
#
#     ax[1].set_title('Scaled data')
#     ax[1].plot(y_scaled, label='scaled data')
#     ax[1].legend()
#     plt.tight_layout()
#
#     ascii_table.print_params(title=None, data={'test': results[test]},
#             invert=True)
#
#     plt.show()
#
#
# def test_filter_data():
#     results = {}
#     test = 'test_filter_data()'
#     results[test] = {}
#     print('\n%s----------%s----------%s'%(BLUE, test, ENDC))
#
# def test_load_and_process():
#     results = {}
#     test = 'test_load_and_process()'
#     results[test] = {}
#     print('\n%s----------%s----------%s'%(BLUE, test, ENDC))
#
#     def load_and_process(db_name, save_location, parameters,
#             interpolated_samples, test, label, default_pass, results,
#             expected_len=None):
#         try:
#             passed = default_pass
#             data = proc.load_and_process(db_name=db_name,
#                 save_location=save_location, parameters=parameters,
#                 interpolated_samples=interpolated_samples)
#         except Exception as e:
#             print('TEST: %s | SUBTEST: %s'%(test, label))
#             print('%s%s%s'%(RED,e,ENDC))
#             passed = not default_pass
#             data = None
#             results[test]['%s'%label] = passed
#             return results, data
#
#         if expected_len is not None:
#             for key in parameters:
#                 if key == 'time':
#                     key = 'cumulative_time'
#                 if len(data[key]) != expected_len:
#                     passed = not default_pass
#                     results[test]['%s'%label] = passed
#                     return results, data
#
#         results[test]['%s'%label] = passed
#         return results, data
#
#     generate_random_traj(steps=147, plot=True)
#
# #CONTINUE FROM HEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEERE
#
#     # cycle through to test with and without interpolated_samples
#     # param doesn't exist
#     # pass without time
#     # pass with time
#     # pass multiple with time
#     loc = 'fake_trajectory'
#     params = ['ee_xyz', 'ideal_trajectory', 'time']
#     interpolated_samples = 50
#     data = load_and_process(
#         db_name=db,
#         save_location=loc,
#         parameters=params,
#         interpolated_samples=interpolated_samples,
#         test=test,
#         label='param does not exist',
#         default_pass=True,
#         results=results,
#         expected_len=interpolated_samples)
#
#     # pass multiple without time
#
#     ascii_table.print_params(title=None, data={'test': results[test]},
#             invert=True)
#
# def test_calc_cartesian_points():
#     results = {}
#     test = 'test_test_calc_cartesian_points()'
#     results[test] = {}
#     print('\n%s----------%s----------%s'%(BLUE, test, ENDC))
#
# def test_two_norm_error():
#     results = {}
#     test = 'test_two_norm_error()'
#     results[test] = {}
#     print('\n%s----------%s----------%s'%(BLUE, test, ENDC))
#
# def generate_random_traj(steps, plot=False):
#     print('Generating fake trajectories...')
#     steps = steps
#     alpha = 0.7
#     ee_xyz = [[np.random.uniform(0.05, 0.2, 1),
#                np.random.uniform(0.05, 0.2, 1),
#                np.random.uniform(0.5, 1.0, 1)]]
#     for ii in range(0,steps):
#         if ii == 0:
#             xx = np.random.uniform(-2, 2, 1)/100
#             yy = np.random.uniform(-2, 2, 1)/100
#             zz = np.random.uniform(-2, 2, 1)/100
#             x = ee_xyz[-1][0] + xx
#             y = ee_xyz[-1][1] + yy
#             z = ee_xyz[-1][2] + zz
#         else:
#             xx = xx * alpha + np.random.uniform(-2, 2, 1)/100 * (1-alpha)
#             yy = yy * alpha + np.random.uniform(-2, 2, 1)/100 * (1-alpha)
#             zz = zz * alpha + np.random.uniform(-2, 2, 1)/100 * (1-alpha)
#             x = ee_xyz[-1][0] + xx
#             y = ee_xyz[-1][1] + yy
#             z = ee_xyz[-1][2] + zz
#         ee_xyz.append([x, y, z])
#
#     ee_xyz = np.squeeze(np.array(ee_xyz))
#
#     alpha = 0.2
#     ideal = np.zeros((steps+1, 3))
#     for ii, val in enumerate(ee_xyz.tolist()):
#         if ii == 0:
#             ideal[0] = val
#         else:
#             ideal[ii][0] = alpha*val[0] + (1-alpha)*ideal[ii-1][0]
#             ideal[ii][1] = alpha*val[1] + (1-alpha)*ideal[ii-1][1]
#             ideal[ii][2] = alpha*val[2] + (1-alpha)*ideal[ii-1][2]
#
#     ideal = np.array(ideal)
#     times = np.ones(steps+1) * 0.03 + np.random.rand(steps+1)/50
#
#     data = {'ee_xyz': ee_xyz, 'ideal_trajectory': ideal, 'time': times}
#     dat.save(data=data, save_location='fake_trajectory', overwrite=True)
#
#     if plot:
#         import matplotlib
#         from mpl_toolkits.mplot3d import axes3d  # pylint: disable=W0611
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.set_title('Generated Trajectory')
#         ax.plot(ee_xyz[:,0], ee_xyz[:,1], ee_xyz[:,2], label='ee_xyz')
#         ax.plot(ideal[:,0], ideal[:,1], ideal[:,2], label='ideal')
#         ax.legend()
#         plt.show()
#
#
# db = 'tests'
