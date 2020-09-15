import matplotlib.pyplot as plt
import numpy as np
import pytest

import abr_analyze.data_processor as proc

from abr_analyze.data_handler import DataHandler
from abr_analyze.utils import random_trajectories


@pytest.mark.parametrize("functions", ([np.sin], [np.sin, np.cos],))
def test_list_to_function(functions, plt):
    samples = 44
    x = np.linspace(0, 6.28, 100)
    time_intervals = x[-1] / 100 * np.ones(100)
    ys = np.array([func(x) for func in functions]).T

    funcs = proc.list_to_function(data=ys, time_intervals=time_intervals)

    x_new = np.linspace(time_intervals[0], x[-1], samples)
    for y, func in zip(ys.T, funcs):
        plt.plot(x, y, "o", label="raw")
        plt.plot(x_new, func(x_new), label="interpolated")
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_interpolate_data_1D(plt):
    data = np.array([1, 2, 3.2, 5.5, 12, 20, 32])
    time_intervals = np.ones(data.shape[0])

    samples = 30
    interp_data = proc.interpolate_data(
        data=data, time_intervals=time_intervals, interpolated_samples=samples
    )

    plt.figure()
    plt.plot(np.cumsum(time_intervals), data, "o", label="raw")
    plt.plot(
        np.linspace(time_intervals[0], np.sum(time_intervals), samples),
        interp_data,
        label="interpolated",
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_interpolate_data_2D(plt):
    data = np.array([[1, 2, 3.2, 5.5, 12, 20, 32], [44, 54, 63, 77, 92, 111, 140]])
    time_intervals = np.ones(data.shape[1])
    samples = 30
    interp_data = proc.interpolate_data(
        data=data.T, time_intervals=time_intervals, interpolated_samples=samples
    )

    x2 = np.linspace(time_intervals[0], np.sum(time_intervals), samples)
    plt.figure()
    for d, y, in zip(data, interp_data.T):
        plt.plot(np.cumsum(time_intervals), d, "o", label="raw")
        plt.plot(x2, y, label="interpolated")
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_scale_data(plt):
    y_low = np.random.uniform(2, 4, 100)
    y_high = np.random.uniform(7, 9, 100)
    y = np.random.uniform(4, 7, 100)
    scale = 1

    y_scaled = proc.scale_data(
        data=y, baseline_low=y_low, baseline_high=y_high, scaling_factor=scale
    )

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title("Raw data")
    plt.plot(y, label="data")
    plt.plot(y_low, label="low baseline")
    plt.plot(y_high, label="high baseline")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title("Scaled data")
    plt.plot(y_scaled, label="scaled data")
    plt.legend()
    plt.tight_layout()
    plt.show()


def load_and_process(interpolated_samples, parameters):
    dat = DataHandler("tests")
    loc = "fake_trajectory"
    steps = 147
    fake_traj_data = random_trajectories.generate(steps=steps, plot=False)
    dat.save(data=fake_traj_data, save_location="fake_trajectory", overwrite=True)

    data = proc.load_and_process(
        db_name="tests",
        save_location=loc,
        parameters=parameters,
        interpolated_samples=interpolated_samples,
    )

    if interpolated_samples is None:
        interpolated_samples = steps

    for key in parameters:
        if key == "time":
            key = "cumulative_time"
        assert len(data[key]) == interpolated_samples


@pytest.mark.parametrize(
    "interpolated_samples, parameters, expected",
    (
        # cycle through to test with and without interpolated_samples
        # param doesn't exist
        (50, ["ee", "ideal", "time"]),
        (None, ["ee", "ideal", "time"]),
    ),
)
def test_load_and_process(interpolated_samples, parameters):
    with pytest.raises(TypeError):
        load_and_process(interpolated_samples, parameters)


@pytest.mark.parametrize(
    "interpolated_samples, parameters",
    (
        # pass without time
        (50, ["ee_xyz"]),
        (None, ["ee_xyz"]),
        # pass with time
        (50, ["ee_xyz", "time"]),
        (None, ["ee_xyz", "time"]),
        # pass multiple with time
        (50, ["ee_xyz", "ideal_trajectory", "time"]),
        (None, ["ee_xyz", "ideal_trajectory", "time"]),
        # pass multiple without time
        (50, ["ee_xyz", "ideal_trajectory"]),
        (None, ["ee_xyz", "ideal_trajectory"]),
    ),
)
def test_load_and_process(interpolated_samples, parameters):
    load_and_process(interpolated_samples, parameters)


def test_calc_cartesion_points():
    db = "tests"
    dat = DataHandler(db)

    class fake_robot_config:
        def __init__(self):
            self.N_JOINTS = 3
            self.N_LINKS = 2

        def Tx(self, name, q):
            assert len(q) == self.N_JOINTS
            return [1, 2, 3]

    robot_config = fake_robot_config()

    # number of time steps
    steps = 10

    # passing in the right dimensions of joint angles
    q = np.zeros((steps, robot_config.N_JOINTS))
    expected_shape = [
        [steps, robot_config.N_JOINTS, 3],
        [steps, robot_config.N_LINKS, 3],
        [steps, 3],
    ]

    data = proc.calc_cartesian_points(robot_config=robot_config, q=q)

    # catch error in the assertion of the functions output's shape
    for ii in range(0, len(expected_shape)):
        for jj in range(0, len(np.array(data[ii]).shape)):
            assert (
                np.array(data[ii]).shape[jj] == expected_shape[ii][jj],
                (
                    "Expected %i Received %i"
                    % (expected_shape[ii][jj], np.asarray(data[ii]).shape[jj])
                ),
            )
