import pytest

import numpy as np
# normally we have a package to reference imports, this is a hack for this
# example to work in a test folder
import os,sys,inspect
current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from abr_analyze.plotting import TrajectoryError
from abr_analyze.data_handler import DataHandler
import generate_random_traj

dat = DataHandler('test')
save_location = 'traj_err_test/session000/run000'
# generate a random trajectory and an ideal
data = generate_random_traj.generate(steps=100)
# generate another trajectory and ideal so we can test passing in a custom ideal
data_alt = generate_random_traj.generate(steps=100)
# save the second ideal to the first data dict
data['alt_traj'] = data_alt['ideal_trajectory']
dat.save(data=data, save_location=save_location, overwrite=True)

@pytest.mark.parametrize('ideal', ((None), ('ideal_trajectory'), ('alt_traj')))
def test_calculate_error(ideal, save_location=save_location):
    dat = DataHandler('test')
    if ideal is None:
        ideal = 'ideal_trajectory'
    fake_data = dat.load(
        parameters=[ideal, 'ee_xyz'], save_location=save_location)
    manual_error = np.linalg.norm(
        (fake_data['ee_xyz'] - fake_data[ideal]), axis=1)

    traj = TrajectoryError(
        db_name='test', time_derivative=0,
        interpolated_samples=None)
    data = traj.calculate_error(
        save_location=save_location, ideal=ideal)

    assert np.array_equal(manual_error, data['error'])


@pytest.mark.parametrize('ideal', ((None), ('ideal_trajectory'), ('alt_traj')))
@pytest.mark.parametrize('save_data', ((True), (False)))
@pytest.mark.parametrize('regen', ((True), (False)))

def test_statistical_error(ideal, save_data, regen, save_location=save_location):
    dat = DataHandler('test')
    if ideal is None:
        # this is done in calculate_error, but we do it here so our manual
        # comparison is loading the same data
        ideal = 'ideal_trajectory'
    fake_data = dat.load(
        parameters=[ideal, 'ee_xyz'], save_location=save_location)
    manual_error = np.linalg.norm(
        (fake_data['ee_xyz'] - fake_data[ideal]), axis=1)

    traj = TrajectoryError(
        db_name='test', time_derivative=0,
        interpolated_samples=None)
    data = traj.statistical_error(
        save_location=save_location.split('/')[0], ideal=ideal, sessions=1,
        runs=1, save_data=save_data, regen=regen)
