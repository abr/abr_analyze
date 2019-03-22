import pytest
import numpy as np

from abr_analyze.nengo_utils import network_utils

@pytest.mark.parametrize('thresh', ((0.001, 0.01, 0.1, 1.0)))
def test_generate_encoders(thresh):

    x = np.arange(0, 2*np.pi, 0.001)
    input_signal = np.vstack([np.sin(x), np.cos(x)]).T

    n_neurons = 100
    encoders = network_utils.generate_encoders(
        n_neurons=n_neurons,
        input_signal=input_signal,
        thresh=thresh)

    # make sure there are no points within threshould
    # distance of each other
    for ii in range(n_neurons):
        assert np.all(np.linalg.norm(encoders - encoders[ii]) > thresh)
    print('Encoders: ', encoders)
