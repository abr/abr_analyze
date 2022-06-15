"""
- a test function is made for every function in the data_handler
- various permutations and cases are tested for each function
- a dictionary is created with entries for every test
  - each test has it's own subtests for the various cases and permutations
  - the subtests are saved with the value of the boolean 'passed'
    which is True if the test passed or failed as expected
  - This is done by placing each subtest in a try except statement.
    This tests if an exception is raised, setting 'passed' based on the
    expected behaviour.
  - Further tests are placed after the try except statement,
    specified for each function (EX: testing if a renamed group exists).
"""
import numpy as np
import pytest

from abr_analyze.data_handler import DataHandler


@pytest.mark.parametrize(
    "data, overwrite",
    (
        # test saving bool
        ({"bool": True}, True),
        # # test overwriting with overwrite=True
        # ({"bool": True}, True),
        # test saving int
        ({"int": int(4)}, True),
        # test saving float
        ({"float": 3.14}, True),
        # test saving complex
        ({"complex": 3.14j}, True),
        # test saving string
        ({"string": "howdy"}, True),
        # test saving tuple
        # NOTE: Warning expected for this test
        ({"tuple": (1, 2, 3)}, True),
        # test saving dict
        # NOTE: Warning expected for this test
        ({"dict": {"test_again": True}}, True),
        # test saving np.array
        ({"np.array": np.ones(2)}, True),
        # test saving None type
        ({"None": None}, True),
    ),
)
def test_save(data, overwrite):
    save_location = "test_saving"
    dat = DataHandler("tests")
    dat.save(data=data, save_location=save_location, overwrite=overwrite)


# test not passing data as dict
def test_save_type_error():
    dat = DataHandler("tests")
    with pytest.raises(TypeError):
        dat.save(data=np.ones(10), save_location="test_saving", overwrite=True)


# test not passing data as dict
def test_save_error():
    dat = DataHandler("tests")
    with pytest.raises(Exception):
        dat.save(data={"bool": True}, save_location="test_saving", overwrite=False)


@pytest.mark.parametrize(
    "parameters, compare_to, key",
    (
        (["test_data"], np.ones(3), "test_data"),
        (["test_data2"], None, "test_data2"),
    ),
)
def test_load(parameters, compare_to, key):
    dat = DataHandler("tests")
    save_location = "test_loading"
    test_data = {"test_data": np.ones(3)}

    dat.save(data=test_data, save_location="test_loading", overwrite=True)

    loaded = dat.load(parameters=parameters, save_location=save_location)

    assert np.all(loaded[key] == compare_to)


def test_load_no_group():
    dat = DataHandler("tests")
    save_location = "not_a_location"

    with pytest.raises(NameError):
        loaded = dat.load(parameters=parameters, save_location=save_location)


def test_get_keys():
    dat = DataHandler("tests")

    # location exists
    keys = dat.get_keys(save_location="test_loading")

    # location doesn't exist
    dat.delete(save_location="fake_location")
    with pytest.raises(KeyError):
        keys = dat.get_keys(save_location="fake_location")


@pytest.mark.parametrize(
    "location, create, compare_to",
    (
        # exists and create False
        ("test_loading", False, True),
        # exists and create True
        ("test_loading", True, True),
        # doesn't exist and create False
        ("fake_location", False, False),
        # doesn't exist and create True
        ("fake_location_now_real", True, True),
    ),
)
def test_group_exists(location, create, compare_to):
    dat = DataHandler("tests")

    exists = dat.check_group_exists(location=location, create=create)
    assert exists == compare_to


@pytest.mark.parametrize(
    "save_location, compare_to",
    (
        # delete group that exists
        ("fake_location_now_real", False),
        # delete dataset that exists
        ("test_saving/bool", False),
    ),
)
def test_delete(save_location, compare_to):
    dat = DataHandler("tests")

    dat.delete(save_location=save_location)

    exists = dat.check_group_exists(location=save_location, create=False)
    assert exists == compare_to


@pytest.mark.parametrize(
    "old_save_location, new_save_location, delete_old," + "compare_to",
    (
        # rename if group exists
        ("test_loading", "test_loading_moved", False, True),
        # delete old save location
        ("test_loading_moved", "test_loading_moved_again", True, False),
    ),
)
def test_rename(old_save_location, new_save_location, delete_old, compare_to):
    dat = DataHandler("tests")
    # save data to rename / move
    dat.save(data={"float": 3.14}, save_location=old_save_location, overwrite=True)
    # make sure the new entry key is available
    dat.delete(save_location=new_save_location)
    # rename to new key
    dat.rename(
        old_save_location=old_save_location,
        new_save_location=new_save_location,
        delete_old=delete_old,
    )

    # check if the old location exists
    exists = dat.check_group_exists(location=old_save_location, create=False)
    assert exists == compare_to

    # check if the new location exists
    exists = dat.check_group_exists(location=new_save_location, create=False)
    assert exists is True


def test_rename_new_save_location_exists():
    old_save_location = "test_rename"
    new_save_location = "test_already_exists"
    with pytest.raises(Exception):
        dat = DataHandler("tests")
        # save data to rename / move
        dat.save(data={"float": 3.14}, save_location=old_save_location)
        # create data at new location
        dat.save(data={"float": 3.14}, save_location=new_save_location, overwrite=True)
        # try to rename data onto existing key
        dat.rename(
            old_save_location=old_save_location,
            new_save_location=new_save_location,
            delete_old=False,
        )


def test_rename_old_save_location_does_not_exist():
    old_save_location = "test_old_does_not_exist"
    new_save_location = "test_new"
    with pytest.raises(KeyError):
        dat = DataHandler("tests")
        # try to read from key that doesn't exist
        dat.rename(
            old_save_location=old_save_location,
            new_save_location=new_save_location,
            delete_old=False,
        )


# try renaming a dataset instead of a group
def test_rename_dataset():
    old_save_location = "test_saving"
    new_save_location = "test_saving_moved"

    dat = DataHandler("tests")
    # save data to rename / move
    dat.save(data={"float": 3.14}, save_location=old_save_location, overwrite=True)
    # make sure the new entry key is available
    dat.delete(save_location=new_save_location)

    dat.rename(
        old_save_location=old_save_location + "/int",
        new_save_location=new_save_location + "/int",
        delete_old=False,
    )

    # check if the old location exists
    exists = dat.check_group_exists(location=old_save_location, create=False)
    assert exists is True

    # check if the new location exists
    exists = dat.check_group_exists(location=new_save_location, create=False)
    assert exists is True
