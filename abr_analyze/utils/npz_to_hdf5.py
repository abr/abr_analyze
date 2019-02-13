from abr_analyze.utils.data_handler import DataHandler
import numpy as np

def convert(npz_loc, db_name, save_location, overwrite=False):
    dat = DataHandler(db_name)
    npz = np.load(npz_loc)
    keys = npz.keys()
    new_data = {}
    for key in keys:
        print(key)
        new_data[key] = npz[key]
    dat.save(data=new_data, save_location=save_location, overwrite=overwrite)
    keys = dat.get_keys(save_location)
    data = dat.load(keys, save_location)
    print(data)
