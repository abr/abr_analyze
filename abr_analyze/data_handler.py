import time
import warnings

import h5py
import numpy as np

from abr_analyze.paths import database_dir as abr_database_dir


class DataHandler:
    """
    Data handler for saving and loading data

    This class is meant to help simplify running automated tests. This is
    specifically helpful when running consecutive runs of learning where it is
    desired to pick up where the last run ended off.

    The naming convention used is as follows:
    - runs are consecutive tests, usually where the previous run is used as
    a starting point for the following run
    - a group of runs is called a session, multiple sessions are used for
      averaging and other statistical purposes like getting confidence
      intervals
    - a test_name is the user specified name of the test being run, which is
      made up of sessions that consist of several runs
    - by default the test_name data is saved in the abr_analyze database_dir.
      the location of this folder is specified in the path.py file

    However, this convention does not have to be used. Any save name can be
    passed in and data will be saved there, and can later be loaded.

    Parameters
    ----------
    db_name: string, Optional (Default: abr_analyze)
        name of the database being used
    """

    def __init__(self, db_name="abr_analyze", database_dir=None, raise_warnings=False):
        self.raise_warnings = raise_warnings
        database_dir = database_dir if database_dir is not None else abr_database_dir
        self.db_loc = f"{database_dir}/{db_name}.h5"

        self.ERRORS = []
        self.db_loc = "%s/%s.h5" % (database_dir, db_name)
        # print('LOADING DATA FROM: ', self.db_loc)
        # Instantiate the database object with the provided path so that it
        # gets created if it does not yet exist
        db = h5py.File(self.db_loc, "a")
        # close the database after each function
        db.close()

    def _save(self, db, save_loc, data, key, overwrite):
        loc_and_key = "%s/%s" % (save_loc, key)
        if self.check_group_exists(loc_and_key):
            if overwrite:
                # if dataset already exists, then overwrite data
                del db[loc_and_key]
            else:
                raise Exception(
                    "Dataset %s already exists in %s" % (key, save_loc)
                    + ": set overwrite=True to overwrite"
                )

        if data is None:
            data = "None"
        try:
            if isinstance(data, tuple):
                data = list(data)
                db[save_loc].create_dataset(key, data=data)
            elif isinstance(data, str):
                dtype = h5py.special_dtype(vlen=str)
                db[save_loc].create_dataset(key, data=data, dtype=dtype)
                # NOTE if catching list of strings can use this
                # data = [data]
                # data_conv = []
                # for i in range(10_000):
                #     data_conv += data
                # print(len(data_conv))
                # longest_word=len(max(data_conv, key=len))
                # print('longest_word=',longest_word)
                #
                # dtype = h5py.special_dtype(vlen=str)
                #
                # arr = np.array(data,dtype='S'+str(longest_word))
                #
                # db[save_loc].create_dataset(key, data=arr, dtype=dtype)
            else:
                db[save_loc].create_dataset(key, data=data)
        except TypeError as e:
            if isinstance(data, dict):
                print(
                    "You can not pass in a list of dicts."
                    + " To save recursive dicts, they must be saved to a dictionary"
                )
            import pdb

            from abr_control.utils import colors

            print(f"\n\n\n{colors.red}Error raised on key: {key}{colors.endc}")
            print(f"{colors.red}{key} has a value of: {data}{colors.endc}")
            print(f"{colors.red}{key} has a type of: {type(data)}{colors.endc}")
            print("Entering pdb for live debugging. Type <exit> to close")
            print("NOTE: key is stored in <key> and value is stored in <data>")
            pdb.set_trace()
            raise e

    def save(
        self,
        data,
        save_location,
        overwrite=False,
        create=True,
        timestamp=False,
    ):
        """
        Saves the data dict passed in to the save_location specified in the
        instantiated database

        Parameters
        ----------
        data: dictionary of lists to save
            instantiate as
                data = {'data1': [], 'data2': []}
            append as
                data['data1'].append(np.copy(data_to_save))
                data['data2'].append(np.copy(other_data_to_save))
        save_location: string, Optional (Default: 'test')
            the group that all of the data will be saved to
        overwrite: boolean, Optional (Default: False)
            determines whether or not to overwrite data if group already exists
            An error gets triggered if the data is being saved to a group
            (folder) that already exists. Setting this to true will ignore that
            and save the data. Data will only get overwritten if the same key
            is used, otherwise the other data in the group will remain
            untouched
        create: boolean, Optional (Default: True)
            determines whether to create the group provided if it does not
            exist, or to warn to the user that it does not
        timestamp: boolean, Optional (Default: True)
            whether to save timestamp with data
        """

        if not isinstance(data, dict):
            raise TypeError("ERROR: data must be a dict, received ", type(data))

        db = h5py.File(self.db_loc, "a")
        if not self.check_group_exists(save_location):
            db.create_group(save_location)

        if timestamp:
            data["timestamp"] = time.strftime("%H:%M:%S")
            data["datestamp"] = time.strftime("%Y/%m/%d")

        for key in data:
            if key is not None:
                if data[key] is None:
                    data[key] = 'None'
                try:
                    try:
                        db[save_location].create_dataset(
                            '%s' % key, data=data[key])

                    except (RuntimeError, ValueError) as e:
                        if overwrite:
                            # if dataset already exists, then overwrite data
                            del db[save_location+'/%s'%key]
                            db[save_location].create_dataset(
                                '%s' % key, data=data[key])
                        else:
                            print(e)
                            raise Exception(
                                'Dataset %s already exists in %s' %
                                (save_location, key) +
                                ': set overwrite=True to overwrite')
                except TypeError as type_error:
                    print('\n\n*****WARNING: SOME DATA DID NOT SAVE*****')
                    print('Trying to save %s to %s' % (key, save_location))
                    print('Received error: %s' %type_error)
                    print('NOTE: HDF5 has no None type and this dataHandler'
                          + ' currently has no test for None entries')
                    print('\n\n')
        db.close()

    def load(self, save_location, parameters=None, recursive=False):
        """
        Accepts a list of parameters and their path to where they are saved in
        the instantiated db, and returns a dictionary of the parameters and their
        values

        PARAMETERS
        ----------
        parameters: list of strings
            ex: ['q', 'dq', 'u', 'adapt']
            if you are unsure about what the keys are for the data saved, you
            can use the get_keys() function to list the keys in a provided
            group path
        save_location: string
            the location to look for data
            EX: 'test_group/test_name/session_num/run_num'
        """
        # check if the group exists
        # print('CHECKING IF GROUP %s EXISTS AT %s' % (parameters, save_location))
        # print('IN DB LOC: ', self.db_loc)
        exists = self.check_group_exists(location=save_location, create=False)

        # if group path does not exist, raise an exception to alert the user
        if exists is False:
            if self.raise_warnings:
                warnings.warn("The path %s does not exist" % (save_location))
            return None

        if parameters is None:
            parameters = self.get_keys(save_location, recursive=recursive)

        # otherwise load the keys
        db = h5py.File(self.db_loc, "a")
        saved_data = {}
        for key in parameters:
            # saved_data[key] = np.array(db.get("%s/%s" % (save_location, key)))
            tmp = db.get("%s/%s" % (save_location, key))
            if not self.is_dataset(f"{save_location}/{key}"):
                tmp = self.load(
                    # parameters=self.get_keys(f"{save_location}/{key}"),
                    save_location=f"{save_location}/{key}",
                    recursive=recursive,
                )
            # TODO:
            # if is a dataset, if recursive load is on then recursively get keys
            # and append them to parameters. Can append to parameters in loop (tested)
            # if self.is_dataset(f"{save_location}/{key}"):
            #     parameters.append(self.get_keys("f{save_location}/{key}", recursive=True))
            elif tmp.dtype == "bool":
                tmp = bool(tmp)
            elif tmp.dtype == "object":
                tmp = tmp.asstr()[()]
                # if not self.is_dataset(f"{save_location}/{key}"):
                #     tmp = tmp.asstr()[()]
                # else:
                #     print(f'test failed looking for dataset type: {tmp}')
            else:
                tmp = np.array(tmp, dtype=tmp.dtype)
            if tmp == 'None':
                tmp = None
            saved_data[key] = tmp

        db.close()

        return saved_data

    def delete(self, save_location):
        """
        Deletes save_location and all contents from instantiated database

        PARAMETERS
        ----------
        save_location: string
            location in the instantiated database to delete
        """
        # TODO: incoprorate KBHit to get user to verify deleting location and
        # print the keys so they are aware of what will be deleted
        try:
            db = h5py.File(self.db_loc, "a")
            del db[save_location]
        except KeyError:
            if self.raise_warnings:
                warnings.warn("No entry for %s" % save_location)

    def rename(self, old_save_location, new_save_location, delete_old=True):
        """
        Renames a group of dataset

        PARAMETERS
        ----------
        old_save_location: string
            save_location to dataset or group to be renamed
        new_save_location: string
            the new save_location to rename old_save_location as
        delete_old: Boolean, Optional(Default:True)
            True to delete old_save_location after renaming
            False to keep both the old and new save_locations
        """
        db = h5py.File(self.db_loc, "a")
        db[new_save_location] = db[old_save_location]
        if delete_old:
            del db[old_save_location]

    def is_dataset(self, save_location):
        """
        Returns true if dataset, False if folder

        save_location: string
            save_location of the group that you want the keys from
            ex: 'my_feature_test/sub_test_group/session000/run003'
        """
        db = h5py.File(self.db_loc, "a")
        try:
            result = isinstance(db[save_location], h5py.Dataset)
        except KeyError as e:
            # key doesn't exist, return False
            result = False
        return result

    def get_keys(self, save_location, recursive=False):
        """
        Takes a path to an hdf5 dataset in the instantiated database and
        returns the keys at that location

        save_location: string
            save_location of the group that you want the keys from
            ex: 'my_feature_test/sub_test_group/session000/run003'
        recursive: bool, Optional (Default: False)
            if True will search through the tree at save_location
            and will return key/sub_key0/..../sub_keyn format keys
            in a list.
            if False will return base level key values in a list.
        """
        db = h5py.File(self.db_loc, "a")
        if not self.check_group_exists(save_location, create=False):
            return [None]
        if isinstance(db[save_location], h5py.Dataset):
            keys = [None]
        else:
            keys = list(db[save_location].keys())

        if recursive:

            def get_recursive(save_location, keys):
                key_hierarchy = []
                num_datasets = 0
                for key in keys:
                    # print('debug: ', key)
                    # print(f'checking {save_location}/{key} to see if dataset')
                    if not isinstance(db[f"{save_location}/{key}"], h5py.Dataset):
                        # print(f"{key} is not a dataset")
                        next_level_keys = list(db[f"{save_location}/{key}"].keys())
                        # print(f"next level keys: {next_level_keys}")
                        for subkey in next_level_keys:
                            key_hierarchy.append(f"{key}/{subkey}")
                        # key_hierarchy = get_recursive(save_location, key_hierarchy)
                        num_datasets += 1
                    else:
                        key_hierarchy.append(f"{key}")
                return key_hierarchy, num_datasets

            num_datasets = 1
            while num_datasets > 0:
                keys, num_datasets = get_recursive(save_location, keys)

        db.close()
        return keys

    def check_group_exists(self, location, create=False):
        """
        Accepts a location in the instantiated database and returns a boolean
        whether it exists. Additionally, the boolean create can be passed in
        that will create the group if it does not exist. Ignores date and
        timestamp

        Parameters
        ----------
        location: string
            The database group that the function checks for,
        create: boolean, Optional (Default:True)
            true: create group if it does not exist
            false: do not create group if it does not exist
        """
        # TODO: should we add check if location is a dataset?
        db = h5py.File(self.db_loc, "a")
        exists = location in db

        if exists is False:
            if create:
                db.create_group(location)
                exists = True
            else:
                exists = False
        db.close()

        return exists

    # TODO: make this function
    def sample_data(self):
        """
        saves every nth data value to save on storage space
        """
        raise Exception("This function is currently not supported")

    # NOTE: these are very control specific, should they be subclassed?
    # TODO: the following functions can probably be cleaned up and shortened
    def last_save_location(
        self,
        session=None,
        run=None,
        test_name="test",
        test_group="test_group",
        create=True,
    ):
        """
        Following the naming structure of save_name/session(int)/run(int) for
        groups, the highest numbered run and session are returned (if not
        specified, otherwise the specified ones are checked)

        If the user sets session or run to None, the function searches the
        specified test_name for the highest numbered run in the
        highest numbered session, otherwise the specified values are used.
        Returns highest numbered run, session and path, unless a run or session
        was specified by the user to use.

        If the user specifies a session, or run that do not exist, the 0th
        session and/or run will be created. However, if the test_name does not
        exist, an exception will be raised to alert the user

        This function is used for both saving and loading, if used for saving
        then it may be desirable to create a group if the provided one does not
        exist. This can be done by setting create to True. The opposite is true
        for loading, where None should be returned if the group does not exist.
        In this scenario create should be set to False. By default, these are
        the settings for the load and save functions

        Parameters
        ----------
        session: int, Optional (Default: None)
            the session number of the current set of runs
            if set to None, then the latest session in the test_name folder
            will be use, based off the highest numbered session
        run: int, Optional (Default: None)
            the run number under which to save data
            if set to None, then the latest run in the test_name/session#
            folder will be used, based off the highest numbered run
        test_name: string, Optional (Default: 'test')
            the folder name that will hold the session and run folders
            the convention is abr_cache_folder/test_name/session#/run#
            The abr cache folder can be found in abr_control/utils/paths.py
        test_group: string, Optional (Default: 'test_group')
            the group that all of the various test_name tests belong to. This
            is helpful for grouping tests together relative to a specific
            feature being tested, a paper that they belong to etc.
        create: Boolean, Optional (Default: True)
            whether to create the group passed in if it does not exist
        """

        self.db = h5py.File(self.db_loc, "a")

        # first check whether the test passed in exists
        exists = self.check_group_exists(
            location="%s/%s/" % (test_group, test_name), create=create
        )

        # if the test does not exist, return None
        if exists is False:
            run = None
            session = None
            location = "%s/%s/" % (test_group, test_name)
            self.db.close()
            return [run, session, location]

        # If a session is provided, check if it exists
        if session is not None:
            # check if the provided session exists before continuing, create it
            # if it does not and create is set to True
            exists = self.check_group_exists(
                location=("%s/%s/session%03d/" % (test_group, test_name, session)),
                create=create,
            )
            # if exists, use the value
            if exists:
                session = "session%03d" % session
            else:
                run = None
                session = None
                location = "%s/%s/" % (test_group, test_name)
                self.db.close()
                return [run, session, location]

        # if not looking for a specific session, check what our highest
        # numbered session is
        elif session is None:
            # get all of the session keys
            session_keys = list(self.db["%s/%s" % (test_group, test_name)].keys())

            if session_keys:
                session = max(session_keys)

            elif create:
                # No session can be found, create it if create is True
                self.db.create_group("%s/%s/session000" % (test_group, test_name))
                session = "session000"

            else:
                run = None
                session = None
                location = "%s/%s/" % (test_group, test_name)
                self.db.close()
                return [run, session, location]

        if run is not None:
            # check if the provided run exists before continuing, create it
            # if it does not and create is set to True
            exists = self.check_group_exists(
                location="%s/%s/%s/run%03d" % (test_group, test_name, session, run),
                create=create,
            )
            # if exists, use the value
            if exists:
                run = "run%03d" % run
            else:
                run = None
                location = "%s/%s/" % (test_group, test_name)
                self.db.close()
                return [run, session, location]

        # usually this will be set to None so that we can start from where we
        # left off in testing, but sometimes it is useful to pick up from
        # a specific run
        elif run is None:
            # get all of the run keys
            run_keys = list(
                self.db["%s/%s/%s" % (test_group, test_name, session)].keys()
            )

            if run_keys:
                run = max(run_keys)

            else:
                run = None

        location = "%s/%s/" % (test_group, test_name)
        if session is not None:
            session = int(session[7:])
            location += "session%03d/" % session
        else:
            location += "%s/" % session
        if run is not None:
            run = int(run[3:])
            location += "run%03d" % run
        else:
            location += "%s/" % run

        self.db.close()
        return [run, session, location]

    def save_run_data(
        self,
        tracked_data,
        session=None,
        run=None,
        test_name="test",
        test_group="test_group",
        overwrite=False,
        create=True,
        timestamp=True,
    ):
        # TODO: currently module does not check whether a lower run or session
        # exists if the user provides a number for either parameter, could lead
        # to a case where user provides run to save as 6, but runs 0-5 do not
        # exist, is it worth adding a check for this?
        """Saves data collected from test trials with
        standard naming convention.

        Uses the naming structure of a session being made up of several runs.
        This allows the user to automate scripts for saving and loading data
        between consecutive runs. These sets of runs are saved in a session, so
        multiple sessions can be run for averaging and other statistical
        purposes

        Parameters
        ----------
        tracked_data: dictionary of lists to save
            instantiate as
                tracked_data = {'data1': [], 'data2': []}
            append as
                tracked_data['data1'].append(np.copy(data_to_save))
                tracked_data['data2'].append(np.copy(other_data_to_save))
        session: int, Optional (Default: None)
            the session number of the current set of runs
            if set to None, then the latest session in the test_name folder
            will be use, based off the highest numbered session
        run: int, Optional (Default: None)
            the run number under which to save data
            if set to None, then the latest run in the test_name/session#
            folder will be used, based off the highest numbered run
        test_name: string, Optional (Default: 'test')
            the folder name that will hold the session and run folders
            the convention is abr_cache_folder/test_name/session#/run#
            The abr cache folder can be found in abr_control/utils/paths.py
        test_group: string, Optional (Default: 'test_group')
            the group that all of the various test_name tests belong to. This
            is helpful for grouping tests together relative to a specific
            feature being tested, a paper that they belong to etc.
        overwrite: boolean, Optional (Default: False)
            determines whether or not to overwrite data if a group / dataset
            already exists
        timestamp: boolean, Optional (Default: True)
            whether to save timestamp with data
        """

        if run is not None:
            run = "run%03d" % run
        if session is not None:
            session = "session%03d" % session
        if session is None or run is None:
            # user did not specify either run or session so we will grab the
            # last entry in the test_name directory based off the highest
            # numbered session and/or run
            [run, session, _] = self.last_save_location(
                session=session,
                run=run,
                test_name=test_name,
                test_group=test_group,
                create=create,
            )

            # if no previous run saved, start saving in run0
            if run is None:
                run = "run000"

        group_path = "%s/%s/%s/%s" % (test_group, test_name, session, run)

        # save the data
        self.save(
            data=tracked_data,
            save_location=group_path,
            overwrite=overwrite,
            create=create,
            timestamp=timestamp,
        )

    def load_run_data(
        self,
        parameters,
        session=None,
        run=None,
        test_name="test",
        test_group="test_group",
        create=False,
    ):
        """
        Loads the data listed in parameters from the group provided

        The path to the group is used as 'test_group/test_name/session/run'
        Note that session and run are ints that from the user end, and are
        later used in the group path as ('run%i'%run) and ('session%i'%session)

        parameters: list of strings
            ex: ['q', 'dq', 'u', 'adapt']
            if you are unsure about what the keys are for the data saved, you
            can use the get_keys() function to list the keys in a provided
            group path
        session: int, Optional (Default: None)
            the session number of the current set of runs
            if set to None, then the latest session in the test_name folder
            will be use, based off the highest numbered session
        run: int, Optional (Default: None)
            the run number under which to save data
            if set to None, then the latest run in the test_name/session#
            folder will be used, based off the highest numbered run
        test_name: string, Optional (Default: 'test')
            the folder name that will hold the session and run folders
            the convention is abr_cache_folder/test_name/session#/run#
            The abr cache folder can be found in abr_control/utils/paths.py
        test_group: string, Optional (Default: 'test_group')
            the group that all of the various test_name tests belong to. This
            is helpful for grouping tests together relative to a specific
            feature being tested, a paper that they belong to etc.
        """
        # if the user doesn'r provide either run or session numbers, the
        # highest numbered run and session are searched for in the provided
        # test_group/test_name location
        if session is None or run is None:
            [run, session, group_path] = self.last_save_location(
                session=session,
                run=run,
                test_name=test_name,
                test_group=test_group,
                create=create,
            )
        else:
            session = "session%03d" % session
            run = "run%03d" % run

        if run is None:
            saved_data = None
        else:
            group_path = "%s/%s/%s/%s" % (test_group, test_name, session, run)

            saved_data = self.load(parameters=parameters, save_location=group_path)

        return saved_data
