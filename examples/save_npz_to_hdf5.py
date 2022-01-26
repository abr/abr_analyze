import numpy as np

from abr_analyze.utils import npz_to_hdf5

# TODO the name of the npz file you are sampling
save_name = "test.npz"
# this is the 'folder' in the database that the data will be saved to
test_name =  'test_0000'

# NOTE We don't want a figure for every dt as that would make more files than we need.
# the gif maker can interpolate the data to a set number of points.
# Alternatively, resave the data sampled every n points and use that to make gifs so
# we don't introduce artifacts with the interpolation
# NOTE you SHOULD sample your data, interpolating data can lead to odd behaviour
sampled_save_name = "sampled_data.npz"
sampling_rate = 40

if sampling_rate is not None:
    print('Saving sampled data to hdf5')
    data = np.load(save_name)
    # print('Original q shape: ', data.keys())
    print(f"Original q shape: {data['q'].shape}")
    # print(data['time'].shape)
    # print(data['ee_xyz'].shape)
    # print(data['ideal_trajectory'].shape)
    # raise Exception
    np.savez_compressed(
        sampled_save_name,
        ideal_trajectory=data['ideal_trajectory'][::sampling_rate],
        ee_xyz=data['ee_xyz'][::sampling_rate],
        q=data['q'][::sampling_rate],
        time=data['time'][::sampling_rate]
    )

    data = np.load(sampled_save_name)
    print(f"Sampled q shape: {data['q'].shape}")

    # Convert npz to hdf5 file
    # NOTE databases get saved to a general location specified in `abr_analyze/paths.py`
    # you can just pass the name of the DB and abr_analyze knows where to look for it
    npz_to_hdf5.convert(
        npz_loc=sampled_save_name,
        db_name="reachkg3_sampled_0000",
        save_location=test_name,  # save location IN the database
        overwrite=True,
    )
else:
    print('Converting raw data to hdf5')
    npz_to_hdf5.convert(
        npz_loc=save_name,
        db_name="reachkg3_0000",
        save_location=test_name,  # save location IN the database
        overwrite=True,
    )
