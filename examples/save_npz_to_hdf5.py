from abr_analyze.utils import npz_to_hdf5
import numpy as np
a = np.ones(11)
b = np.zeros(4)
save_name = 'test.npz'
np.savez_compressed(save_name, a=a, c=b)
npz_to_hdf5.convert(
        npz_loc=save_name,
        db_name='abr_analyze',
        save_location='my_converted_data/test1',
        overwrite=True)
