import numpy as np

from abr_analyze.utils import npz_to_hdf5

# TODO the name of the npz file you are sampling
save_name = "test.npz"
# this is the 'folder' in the database that the data will be saved to
test_name =  'test_0000'

examples_db()
a = np.ones(11)
b = np.zeros(4)
save_name = "test.npz"
np.savez_compressed(save_name, a=a, c=b)
npz_to_hdf5.convert(
    npz_loc=save_name,
    db_name="abr_analyze_examples",
    save_location="my_converted_data/test1",
    overwrite=True,

