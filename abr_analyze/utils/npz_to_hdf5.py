"""
simple function for taking in an npz file location and saving it's keys and
corresponding data to a specified save location in the database
"""
import numpy as np

from abr_analyze.data_handler import DataHandler


def convert(npz_loc, db_name, save_location, overwrite=False):
    """
    accepts a npz file location and saves its data to the database at the
    specified save_location

    PARAMETERS
    ----------
    npz_loc: string
        location and name of the npz file
    db_name: string
        database to save data to
    save_location: string
        save location in the database
    overwrite: boolean, Optional (Default: False)
        True to overwrite previous data
        NOTE this will be triggered if data is being saved to the same hdf5
        group (folder), not necessarily the same key. In this case you will
        need to set it to True. Other data will not be erased, only data with
        the same keys will be overwritten
    """
    dat = DataHandler(db_name)
    npz = np.load(npz_loc)
    keys = npz.keys()
    new_data = {}
    for key in keys:
        # print(key)
        new_data[key] = npz[key]
    dat.save(data=new_data, save_location=save_location, overwrite=overwrite)
    keys = dat.get_keys(save_location)
    data = dat.load(keys, save_location)
    # print(data)
