'''
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
'''
import numpy as np
from abr_analyze.data_handler import DataHandler
from abr_analyze.utils import ascii_table

BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW= '\033[93m'
RED = '\033[91m'
ENDC = '\033[0m'

# test saving data
def test_save():
    results = {}
    test = 'test_save()'
    results[test] = {}
    print('\n%s----------%s----------%s'%(BLUE, test, ENDC))

    def save(data, save_location, test, label,
             overwrite, default_pass, results):
        try:
            passed = default_pass
            dat.save(data=data,
                     save_location=save_location,
                     overwrite=overwrite)
        except Exception as e:
            print('TEST: %s | SUBTEST: %s'%(test, label))
            print('%s%s%s'%(RED,e,ENDC))
            passed = not default_pass
        results[test]['%s'%label] = passed
        return results

    # not passing data as dict
    not_dict = np.ones(10)
    results = save(
        data=not_dict,
        save_location='test_saving',
        test=test,
        label='data not dict',
        overwrite=True,
        default_pass=False,
        results=results)
    # test saving bool
    d_bool = {'bool': True}
    results = save(
        data=d_bool,
        save_location='test_saving',
        test=test,
        label='bool type',
        overwrite=True,
        default_pass=True,
        results=results)
    # test overwriting with overwrite=False
    results = save(
        data=d_bool,
        save_location='test_saving',
        test=test,
        label='overwrite with overwrite=False',
        overwrite=False,
        default_pass=False,
        results=results)
    # test overwriting with overwrite=True
    results = save(
        data=d_bool,
        save_location='test_saving',
        test=test,
        label='overwrite with overwrite=True',
        overwrite=True,
        default_pass=True,
        results=results)
    # test saving int
    d_int = {'int': int(4)}
    results = save(
        data=d_int,
        save_location='test_saving',
        test=test,
        label='int type',
        overwrite=True,
        default_pass=True,
        results=results)
    # test saving float
    d_float = {'float': 3.14}
    results = save(
        data=d_float,
        save_location='test_saving',
        test=test,
        label='float type',
        overwrite=True,
        default_pass=True,
        results=results)
    # test saving complex
    d_complex = {'complex': 3.14J}
    results = save(
        data=d_complex,
        save_location='test_saving',
        test=test,
        label='complex type',
        overwrite=True,
        default_pass=True,
        results=results)
    # test saving string
    d_string = {'string': 'howdy'}
    results = save(
        data=d_string,
        save_location='test_saving',
        test=test,
        label='string_type',
        overwrite=True,
        default_pass=True,
        results=results)
    # test saving tuple
    d_tuple = {'tuple': ('what', 'is', 'a', 'tuple', '?')}
    results = save(
        data=d_tuple,
        save_location='test_saving',
        test=test,
        label='tuple type',
        overwrite=True,
        default_pass='True if WARNING printed above',
        results=results)
    # test saving dict
    d_dict = {'dict':{'test_again': True}}
    results = save(
        data=d_dict,
        save_location='test_saving',
        test=test,
        label='dict type',
        overwrite=True,
        default_pass='True if WARNING printed above',
        results=results)
    # test saving np.array
    d_np_array = {'np.array': np.ones(2)}
    results = save(
        data=d_np_array,
        save_location='test_saving',
        test=test,
        label='np.array type',
        overwrite=True,
        default_pass=True,
        results=results)
    # test saving None type
    d_None = {'None': None}
    results = save(
        data=d_None,
        save_location='test_saving',
        test=test,
        label='None type',
        overwrite=True,
        default_pass=True,
        results=results)
    ascii_table.print_params(title=None, data={'test': results[test]},
            invert=True)

# test loading data
def test_load():
    results = {}
    test = 'test_load()'
    results[test] = {}
    print('\n%s----------%s----------%s'%(BLUE, test, ENDC))

    def load(parameters, save_location, test, label,
             default_pass, results, compare_to, key):
        try:
            passed = default_pass
            loaded = dat.load(parameters=parameters,
                              save_location=save_location)
        except Exception as e:
            print('TEST: %s | SUBTEST: %s'%(test, label))
            print('%s%s%s'%(RED,e,ENDC))
            passed = not default_pass
            results[test]['%s'%label] = passed
            return results

        if compare_to is not None:
            if np.array(loaded[key]).all() != np.array(compare_to).all():
                passed = not default_pass
        results[test]['%s'%label] = passed
        return results

    test_data = {'test_data': np.ones(3)}
    dat.save(
        data=test_data,
        save_location='test_loading',
        overwrite=True)
    # test loading group and key exist
    results = load(
        parameters=['test_data'],
        save_location='test_loading',
        test=test,
        label='loading correctly',
        default_pass=True,
        results=results,
        compare_to=test_data['test_data'],
        key='test_data')
    # test loading group exists, key doesn't
    # this should not fail, simply return a blank values entry
    results = load(
        parameters=['test_data2'],
        save_location='test_loading',
        test=test,
        label='key doesn\'t exist',
        default_pass=True,
        results=results,
        compare_to=None,
        key='test_data2')
    # test loading group doesn't exist
    results = load(
        parameters=['test_data'],
        save_location='not_a_location',
        test=test,
        label='group doesn\'t exist',
        default_pass=False,
        results=results,
        compare_to=None,
        key=None)
    ascii_table.print_params(title=None, data={'test': results[test]},
            invert=True)

# test getting keys
def test_get_keys():
    results = {}
    test = 'test_get_keys()'
    results[test] = {}
    print('\n%s----------%s----------%s'%(BLUE, test, ENDC))

    def get_keys(save_location, test, label,
                 default_pass, results):
        try:
            passed = default_pass
            keys = dat.get_keys(
                save_location=save_location)
        except Exception as e:
            print('TEST: %s | SUBTEST: %s'%(test, label))
            print('%s%s%s'%(RED,e,ENDC))
            passed = not default_pass

        results[test]['%s'%label] = passed
        return results

    # location exists
    results = get_keys(
        save_location='test_loading',
        test=test,
        label='location exists',
        default_pass=True,
        results=results)

    # location doesn't exists
    results = get_keys(
        save_location='fake location',
        test=test,
        label='location doesn\'t exist',
        default_pass=False,
        results=results)
    ascii_table.print_params(title=None, data={'test': results[test]},
            invert=True)


# test checking if group exists
def test_group_exists():
    results = {}
    test = 'test_group_exists()'
    results[test]  = {}
    print('\n%s----------%s----------%s'%(BLUE, test, ENDC))

    def group_exists(location, create, test, label,
                     default_pass, results, compare_to):
        try:
            passed = default_pass
            exists = dat.check_group_exists(
                location=location,
                create=create)
        except Exception as e:
            print('TEST: %s | SUBTEST: %s'%(test, label))
            print('%s%s%s'%(RED,e,ENDC))
            passed = not default_pass
            results[test]['%s'%label] = passed
            return results

        if exists != compare_to:
            passed = not default_pass
        results[test]['%s'%label] = passed
        return results

    # exists and create False
    results = group_exists(
        location='test_loading',
        create=False,
        test=test,
        label='exists and create=False',
        default_pass=True,
        results=results,
        compare_to=True)
    # exists and create True
    results = group_exists(
        location='test_loading',
        create=True,
        test=test,
        label='exists and create=True',
        default_pass=True,
        results=results,
        compare_to=True)
    # doesn't exist and create False
    results = group_exists(
        location='fake_location',
        create=False,
        test=test,
        label='doesn\'t exist and create=False',
        default_pass=True,
        results=results,
        compare_to=False)
    # doesn't exist and create True
    results = group_exists(
        location='fake_location_now_real',
        create=True,
        test=test,
        label='exists and create=True',
        default_pass=True,
        results=results,
        compare_to=True)
    ascii_table.print_params(title=None, data={'test': results[test]},
            invert=True)

# test deleting data
def test_delete():
    results = {}
    test = 'test_delete()'
    results[test]  = {}
    print('\n%s----------%s----------%s'%(BLUE, test, ENDC))

    def delete(save_location, test, label,
               default_pass, results, compare_to):
        try:
            passed = default_pass
            dat.delete(save_location=save_location)
        except Exception as e:
            print('TEST: %s | SUBTEST: %s'%(test, label))
            print('%s%s%s'%(RED,e,ENDC))
            passed = not default_pass
            results[test]['%s'%label] = passed
            return results

        exists = dat.check_group_exists(
            location=save_location,
            create=False)
        if compare_to is not None:
            if exists != compare_to:
                passed = not default_pass

        results[test]['%s'%label] = passed
        return results

    # delete group that exists
    results = delete(
        save_location='fake_location_now_real',
        test=test,
        label='delete group that exists',
        default_pass=True,
        results=results,
        compare_to=False)
    # delete group that  doesn't exist
    results = delete(
        save_location='fake_location',
        test=test,
        label='delete group that doesn\'t exists',
        default_pass=False,
        results=results,
        compare_to=None)
    # delete dataset that exists
    results = delete(
        save_location='test_saving/bool',
        test=test,
        label='delete dataset that exists',
        default_pass=True,
        results=results,
        compare_to=None)
    # delete dataset that doesn't exists
    results = delete(
        save_location='test_saving/bool',
        test=test,
        label='delete dataset that doesn\'t exists',
        default_pass=False,
        results=results,
        compare_to=None)
    ascii_table.print_params(title=None, data={'test': results[test]},
            invert=True)


# test renaming data
def test_rename():
    results = {}
    test = 'test_rename()'
    results[test]  = {}
    print('\n%s----------%s----------%s'%(BLUE, test, ENDC))

    def rename(old_save_location, new_save_location,
               delete_old, test, label, default_pass, results,
               compare_to):
        try:
            passed = default_pass
            dat.rename(
                       old_save_location=old_save_location,
                       new_save_location=new_save_location,
                       delete_old=delete_old)
        except Exception as e:
            print('TEST: %s | SUBTEST: %s'%(test, label))
            print('%s%s%s'%(RED,e,ENDC))
            passed = not default_pass
            results[test]['%s'%label] = passed
            return results

        # check if the old location exists based and compare to
        # the expected value
        if compare_to is not None:
            exists = dat.check_group_exists(
                location=old_save_location,
                create=False)
            if exists != compare_to:
                passed = not default_pass
                results[test]['%s'%label] = passed
                return results

        # check if the new location exists
        exists = dat.check_group_exists(
            location=new_save_location,
            create=False)
        if exists is not True:
            passed = not default_pass
        results[test]['%s'%label] = passed
        return results

    # rename if group exists
    results = rename(
        old_save_location='test_loading',
        new_save_location='test_loading_moved',
        delete_old=False,
        test=test,
        label='Move group that exists, do not delete old',
        default_pass=True,
        results=results,
        compare_to=True)
    # rename to group that already exists
    results = rename(
        old_save_location='test_loading',
        new_save_location='test_loading_moved',
        delete_old=False,
        test=test,
        label='Move group that exists, to location that exists',
        default_pass=False,
        results=results,
        compare_to=None)
    # rename if group doesn't exist
    results = rename(
        old_save_location='fake_location',
        new_save_location='fake_location_moved',
        delete_old=False,
        test=test,
        label='Move group that does not exists',
        default_pass=False,
        results=results,
        compare_to=False)
    # try renaming a dataset instead of a group
    results = rename(
        old_save_location='test_saving/int',
        new_save_location='test_saving_moved/int',
        delete_old=False,
        test=test,
        label='Move dataset that exists',
        default_pass=True,
        results=results,
        compare_to=True)
    # delete old save location
    results = rename(
        old_save_location='test_loading_moved',
        new_save_location='test_loading_moved_again',
        delete_old=True,
        test=test,
        label='Move group that exists, delete old',
        default_pass=True,
        results=results,
        compare_to=False)

    # delete the moved groups so the test doesn't fail the next time
    dat.delete(save_location='test_loading_moved_again')
    dat.delete(save_location='test_saving_moved')
    ascii_table.print_params(title=None, data={'test': results[test]},
            invert=True)


db = 'tests'
dat = DataHandler(db)
test_save()
test_load()
test_get_keys()
test_group_exists()
test_delete()
test_rename()
