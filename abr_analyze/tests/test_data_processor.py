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
    time_intervals = x[-1]/100 * np.ones(100)
    ys = np.array([func(x) for func in functions]).T

    funcs = proc.list_to_function(data=ys, time_intervals=time_intervals)

    x_new = np.linspace(time_intervals[0], x[-1], samples)
    for y, func in zip(ys.T, funcs):
        plt.plot(x, y, 'o', label='raw')
        plt.plot(x_new, func(x_new), label='interpolated')
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_interpolate_data_1D(plt):
    data = np.array([1, 2, 3.2, 5.5, 12, 20, 32])
    time_intervals = np.ones(data.shape[0])

    samples = 30
    interp_data = proc.interpolate_data(
        data=data,
        time_intervals=time_intervals,
        interpolated_samples=samples)

    plt.figure()
    plt.plot(np.cumsum(time_intervals), data, 'o', label='raw')
    plt.plot(np.linspace(time_intervals[0], np.sum(time_intervals), samples),
                         interp_data, label='interpolated')
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_interpolate_data_2D(plt):
    data = np.array([[1, 2, 3.2, 5.5, 12, 20, 32],
                     [44, 54, 63, 77, 92, 111, 140]])
    time_intervals = np.ones(data.shape[1])
    samples = 30
    interp_data = proc.interpolate_data(
        data=data.T,
        time_intervals=time_intervals,
        interpolated_samples=samples)

    x2 = np.linspace(time_intervals[0], np.sum(time_intervals), samples)
    plt.figure()
    for d, y, in zip(data, interp_data.T):
        plt.plot(np.cumsum(time_intervals), d, 'o', label='raw')
        plt.plot(x2, y, label='interpolated')
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_scale_data(plt):
    y_low = np.random.uniform(2, 4, 100)
    y_high = np.random.uniform(7, 9, 100)
    y = np.random.uniform(4, 7, 100)
    scale = 1

    y_scaled = proc.scale_data(
        data=y,
        baseline_low=y_low,
        baseline_high=y_high,
        scaling_factor=scale)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('Raw data')
    plt.plot(y, label='data')
    plt.plot(y_low, label='low baseline')
    plt.plot(y_high, label='high baseline')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title('Scaled data')
    plt.plot(y_scaled, label='scaled data')
    plt.legend()
    plt.tight_layout()
    plt.show()


# def test_filter_data():
#     results = {}
#     test = 'test_filter_data()'
#     results[test] = {}
#     print('\n%s----------%s----------%s'%(BLUE, test, ENDC))


def generate_random_traj(dat, steps, plot=False):
    steps = steps
    alpha = 0.7
    ee_xyz = [[np.random.uniform(0.05, 0.2, 1),
               np.random.uniform(0.05, 0.2, 1),
               np.random.uniform(0.5, 1.0, 1)]]

    for ii in range(0,steps):

        if ii == 0:
            xx = np.random.uniform(-2, 2, 1)/100
            yy = np.random.uniform(-2, 2, 1)/100
            zz = np.random.uniform(-2, 2, 1)/100
            x = ee_xyz[-1][0] + xx
            y = ee_xyz[-1][1] + yy
            z = ee_xyz[-1][2] + zz

        else:
            xx = xx * alpha + np.random.uniform(-2, 2, 1)/100 * (1-alpha)
            yy = yy * alpha + np.random.uniform(-2, 2, 1)/100 * (1-alpha)
            zz = zz * alpha + np.random.uniform(-2, 2, 1)/100 * (1-alpha)
            x = ee_xyz[-1][0] + xx
            y = ee_xyz[-1][1] + yy
            z = ee_xyz[-1][2] + zz

        ee_xyz.append([x, y, z])

    ee_xyz = np.squeeze(np.array(ee_xyz))

    alpha = 0.2
    ideal = np.zeros((steps+1, 3))
    for ii, val in enumerate(ee_xyz.tolist()):

        if ii == 0:
            ideal[0] = val

        else:
            ideal[ii][0] = alpha*val[0] + (1-alpha)*ideal[ii-1][0]
            ideal[ii][1] = alpha*val[1] + (1-alpha)*ideal[ii-1][1]
            ideal[ii][2] = alpha*val[2] + (1-alpha)*ideal[ii-1][2]

    ideal = np.array(ideal)
    times = np.ones(steps+1) * 0.03 + np.random.rand(steps+1)/50

    data = {'ee_xyz': ee_xyz, 'ideal_trajectory': ideal, 'time': times}
    dat.save(data=data, save_location='fake_trajectory', overwrite=True)

    if plot:
        from mpl_toolkits.mplot3d import axes3d  # pylint: disable=W0611
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('Generated Trajectory')
        ax.plot(ee_xyz[:,0], ee_xyz[:,1], ee_xyz[:,2], label='ee_xyz')
        ax.plot(ideal[:,0], ideal[:,1], ideal[:,2], label='ideal')
        ax.legend()
        plt.show()


def test_load_and_process():
    dat = DataHandler('tests')

    interpolated_samples = 50
    parameters = ['ee_xyz', 'ideal_trajectory', 'time']
    data = proc.load_and_process(
        db_name='tests',
        save_location='fake_trajectory',
        parameters=parameters,
        interpolated_samples=interpolated_samples,
        )

    for key in parameters:
        if key == 'time':
            key = 'cumulative_time'
        assert len(data[key]) == interpolated_samples


    generate_random_traj(dat, steps=147, plot=True)

    # cycle through to test with and without interpolated_samples
    # param doesn't exist
    # pass without time
    # pass with time
    # pass multiple with time


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
