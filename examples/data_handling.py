'''
This script shows how to use the hdf5 database structure, including saving,
loading, renaming, and deleting data
'''
import numpy as np

from abr_analyze import DataHandler
from download_examples_db import check_exists as examples_db


examples_db()
save_location = 'data_handling'
recorded_time = np.linspace(0,1,100)
recorded_data = np.random.rand(100,3)
data_dict = {'trajectory': recorded_data, 'time': recorded_time}

# instantiate a database with your desired name
dat = DataHandler(db_name='abr_analyze_examples')

# save our data
dat.save(
    data=data_dict,
    save_location=save_location,
    overwrite=True)

# load our data
# we can specify what parameters to load
data = dat.load(
    parameters=['trajectory', 'time'],
    save_location=save_location)
trajectory = data['trajectory']
time = data['time']

# we can rename our save_location as well
new_save_location = 'data_handling_rename'
dat.rename(
    old_save_location=save_location,
    new_save_location=new_save_location,
    delete_old=True)

# if we don't know the parameters, or want to load all of them and want to
# avoid writing out the entire list we can get all the keys at the save
# location
keys = dat.get_keys(save_location=new_save_location)
data = dat.load(
    parameters=keys,
    save_location=new_save_location)

# delete data we don't need anymore
dat.delete(save_location='%s/time'%new_save_location)

# we can also delete the entire group if we want
dat.delete(save_location=new_save_location)
