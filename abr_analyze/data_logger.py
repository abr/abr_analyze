"""
Some helper functions that work with abr_analyze to allow for searchable result and parameter saving.
Requires:
    - a unique name for script generating results
    - results in a dict
    - parameters used to generate results in a dict
    - a DataHandler() instance to save to

Allows for a list of constant parameters to be passed in to load_results() and will return a dictionary
of results that were generating with parameters that match the set of constant params.

Also generates unique name for each results using its varied parameters, and gives a dict of key, value
pairs for all parameters that are constant between loaded tests.

See _example() below for simple use case
"""
import copy
import hashlib
import json
import sys
from itertools import product

import numpy as np

from abr_analyze import DataHandler

blue = "\033[94m"
endc = "\033[0m"
green = "\033[92m"
red = "\033[91m"


def gen_hash_name(params):
    hash_name = hashlib.sha256(str(params).replace(" ", "").encode("utf-8")).hexdigest()
    return hash_name


def print_nested(d, indent=0, return_val=False):
    """
    Pretty printing of nested dictionaries
    """
    if return_val:
        full_print = ""
    for key, value in d.items():
        if isinstance(value, dict):
            line = "\t" * indent + str(key) + ": "
            if return_val:
                full_print += line
            else:
                print(line)
            if return_val:
                nested_line = print_nested(value, indent + 1, return_val=return_val)
                full_print += nested_line
            else:
                print_nested(value, indent + 1)
        else:
            line = "\t" * indent + str(key) + f": {value}"
            if return_val:
                full_print += line
            else:
                print(line)

    if return_val:
        return full_print


def dict_nested2str(d, indent=4, _recursive_call=False):
    str_dict = ""
    if _recursive_call:
        internal_indent = indent
    else:
        internal_indent = 0
    # print('internal: ', internal_indent)
    for key, value in d.items():
        if isinstance(value, dict):
            str_dict += "\n" + " " * internal_indent + str(key) + ": "
            # str_dict += str(key) + ": "
            # str_dict += '-woah-' + str(value)
            str_dict += dict_nested2str(value, indent * 2, _recursive_call=True)
        else:
            str_dict += "\n" + " " * internal_indent + str(key) + f": {value}"
    return str_dict


def gen_lookup_table(db_name, db_folder):
    dat = DataHandler(db_name=db_name, database_dir=db_folder)

    hashes = dat.load(save_location="params", parameters=dat.get_keys("params"))
    lookup = {}
    for hash_id in hashes:
        params = dat.load(
            save_location=f"params/{hash_id}",
            parameters=dat.get_keys(f"params/{hash_id}", recursive=True),
        )
        for key, val in params.items():
            if key not in lookup:
                lookup[key] = {str(val): [hash_id]}
            elif str(val) not in lookup[key].keys():
                lookup[key][str(val)] = [hash_id]
            elif str(val) in lookup[key].keys():
                lookup[key][str(val)].append(hash_id)
    return lookup


# def add_parameters(missing_param_dict, db_name, db_folder=None):
#     dat = DataHandler(
#         db_name=db_name,
#         database_dir=db_folder
#     )
#     hash_ids = dat.load(save_location='params')


def find_constant_and_variable_parameters(dat, saved_exp_hashes, parameter_stems=None):
    """
    Input a DataHandler object and list of experiment hashes and will
    return a dictionary of constant parameters and a list of keys with
    that differ between any of the experiments. The differing values
    are to be used as legend keys to differentiate between experiments.
    The constant parameters can be printed alongside the figure.

    Returns dictionary of constant parameters between all saved_exp_hashes
    and a list of keys that are variable between saved_exp_hashes. The constants
    can be printed as text in a plot, and the variable keys can be used
    to autogenerate legends to show differing values between experiments.


    Saving format
    params  >hash_0 >parameter in json format
            >hash_1 >parameter in json format
            >hash_2 >parameter in json format

    results >script_0   >hash_0 >results_dict
                        >hash_2 >results_dict
            >script_1   >hash_1 >results_dict

    Parameters
    ----------
    dat: instantiated DataHandler
    saved_exp_hashes: list of strings
        Experiment hashes to compare keys.
    parameter_stems: list of strings, Optional (Default: [''])
        List of root keys. The subkeys of these values will be
        compared. Leave parameter stems as [''] if there are no
        nested dictionaries.
    """
    if isinstance(parameter_stems, str):
        parameter_stems = [parameter_stems]
    # TODO might not need this now that can get nested keys
    if parameter_stems is None:
        parameter_stems = [""]

    final_legend = []
    final_constants = {}

    # for group_name in parameter_stems:
    for ee, exp_hash in enumerate(saved_exp_hashes):
        # print(f"ee: {ee}")
        # keys = dat.get_keys(f"params/{exp_hash}/{group_name}")
        # track any differing values keys'
        legend_keys = []
        if ee == 0:
            # base_parameters = dat.load(save_location=f"params/{exp_hash}/{group_name}", parameters=keys)
            base_parameters = dat.load(
                save_location=f"params/{exp_hash}", recursive=True
            )
        else:
            # new_parameters = dat.load(save_location=f"params/{exp_hash}/{group_name}", parameters=keys)
            new_parameters = dat.load(
                save_location=f"params/{exp_hash}", recursive=True
            )
            # temporary storage of differing keys to add to legend keys
            differing_keys = []
            for key in base_parameters:
                if isinstance(base_parameters[key], (list, np.ndarray)):
                    try:
                        # if (base_parameters[key] != new_parameters[key]).any():
                        #     differing_keys.append(key)
                        if (
                            np.asarray(base_parameters[key]).shape
                            != np.asarray(new_parameters[key]).shape
                        ):
                            differing_keys.append(key)
                        elif type(base_parameters[key]) != type(new_parameters[key]):
                            differing_keys.append(key)
                        elif (base_parameters[key] != new_parameters[key]).any():
                            differing_keys.append(key)

                    except AttributeError as e:
                        print(
                            f"Got AttributeError on {key} who's value is:\n{base_parameters[key]}"
                        )
                        print(f"Or possibly from const params:\n{new_parameters[key]}")
                        raise e

                else:
                    if key not in new_parameters.keys():
                        new_parameters[key] = None
                    if type(base_parameters[key]) != type(new_parameters[key]):
                        differing_keys.append(key)
                    elif base_parameters[key] != new_parameters[key]:
                        differing_keys.append(key)

            # add missing keys directly to legend keys
            for key in new_parameters:
                if key not in base_parameters.keys():
                    # legend_keys.append(f"{group_name}/{key}")
                    legend_keys.append(f"{key}")

            # remove differing keys from base parameters, only leaving common ones
            for key in differing_keys:
                base_parameters.pop(key)
                # legend_keys.append(f"{group_name}/{key}")
                legend_keys.append(f"{key}")

    return base_parameters, legend_keys


def find_experiments_that_match_constants(dat, saved_exp_hashes, const_params):
    """
    Input an instantiated DataHandler, a list of experiment hashes to compare,
    and a dictionary of parameters and values.

    Returns a list of experiment hashes that match the constant parameters.

    Parameters
    ----------
    dat: instantiated DataHandler
    saved_exp_hashes: list of strings
        Experiment hashes to compare keys.
    const_params: dict
        Dictionary of parameter values used to find experiments that contain them
    """
    matches = []
    print("EXP HASHES: ", saved_exp_hashes)
    for exp_hash in saved_exp_hashes:
        data = dat.load(
            save_location=f"params/{exp_hash}", parameters=const_params.keys()
        )
        # count the number of different key: value pairs, if zero save the hash
        # since looking for experiments with matching parameters
        num_diff = 0
        for param_key in const_params.keys():
            # print("param key: ", param_key)
            # print(data)
            if data[param_key] is not None and isinstance(
                data[param_key], (list, np.ndarray)
            ):
                try:
                    # print(param_key)
                    if (
                        np.asarray(data[param_key]).shape
                        != np.asarray(const_params[param_key]).shape
                    ):
                        num_diff += 1
                    elif (data[param_key] != const_params[param_key]).any():
                        num_diff += 1
                except AttributeError as e:
                    print(
                        f"Got AttributeError on {param_key} who's value is:\n{data[param_key]}"
                    )
                    print(f"Or possibly from const params:\n{const_params[param_key]}")
                    raise e
            elif isinstance(data[param_key], dict):
                raise NotImplementedError(
                    f"{red}Currently not able to pass nested dict in as const_params{endc}"
                    + f"{red}\n To pass in nested dict items, pass the nested keys in{endc} "
                    + f"{red}separated by as slash '/'{endc}"
                )
            else:
                if data[param_key] != const_params[param_key]:
                    num_diff += 1
        if num_diff == 0:
            matches.append(exp_hash)
    return matches


def get_common_experiments(
    script_name, dat, const_params=None, ignore_keys=None, saved_exp_hashes=None
):
    """
    Input the script name to load results from and the desired set of constant
    parameters between experiments. If const_params is left as None, all parameters
    will be compared to find the constants. Will search which experiments have saved
    results for the specified script, and will search their parameters to find
    which ones match the values of constant_params.

    Returns list of experiment hashes that match constant params, a dictionary
    of the constant parameters, and a list of keys of parameters that differ

    Parameters
    ----------
    script_name: string
        unique id for the script that generated the results to be loaded
    ignore_keys: list of strings, Optinoal (Default: None)
        keys to ignore for legend values
    """

    # Load the hashes of all experiments that have been run for this script
    if saved_exp_hashes is None:
        saved_exp_hashes = dat.get_keys(f"results/{script_name}")
        print(
            f"{len(saved_exp_hashes)} experiments found with results from {script_name}"
        )

    if const_params is not None:
        # Get all experiment id's that match a set of key value pairs
        print(f"Searching for results with matching parameters to:")
        print_nested(const_params)
        saved_exp_hashes = find_experiments_that_match_constants(
            dat, saved_exp_hashes, const_params
        )
        print(f"{len(saved_exp_hashes)} experiments found with matching parameters")
        print(saved_exp_hashes)

    # Get a dictionary of common values and a list of keys for differing values
    # to use in the auto legend
    all_constants, all_variable = find_constant_and_variable_parameters(
        dat, saved_exp_hashes, parameter_stems=["llp", "data", "general", "ens_args"]
    )
    if ignore_keys is not None:
        if isinstance(ignore_keys, str):
            ignore_keys = [ignore_keys]

        for key in ignore_keys:
            if key in all_variable:
                all_variable.remove(key)

    return saved_exp_hashes, all_constants, all_variable


def load_results(
    script_name,
    result_keys,
    dat,
    # db_name,
    # db_folder=None,
    const_params=None,
    saved_exp_hashes=None,
    ignore_keys=None,
):
    """
    Input unique script id that generated results, the values of parameters to
    hold constant between expriments, and the result keys to load. Returns a
    dictionary of experiments who match the desired constant parameters, and
    have results from the set script name.

    Returns dictionary in the format:
    {
        experiment_hash1:
            {
                'name': human readable id (string_of_variable_parameters key:value | ...),
                'results':
                    {
                        'result_key1': value,
                        ...
                        'result_keyn': value
                    }
            },
        ...,
        experiment_hashn:
            {
            ...
            },

        'const_params':
            {
                'const_param1': value,
                ...
                'const_paramn': value
            },
        'variable_params': [variable_param1, ..., variable_paramn]
    }

    The idea is to use the <name> in plot legends since they contain the key value pairs
    of parameters that differ between all experiments. If manual string formatting is desired
    for the name, the <variable_params> keys are also provided so they can be loaded from
    an instantiated DataHandler at <params/experiment_hash>.
    """
    # assert not (saved_exp_hashes is None and const_params is None), (
    #     "Have to provide either experiment hashes to load,"
    #     +" or a dict of constant parameters to find experiments"
    #     +" with matching values"
    # )

    saved_exp_hashes, all_constants, all_variable = get_common_experiments(
        script_name=script_name,
        dat=dat,
        const_params=const_params,
        ignore_keys=ignore_keys,
        saved_exp_hashes=saved_exp_hashes,
    )

    results = {}
    results["const_params"] = all_constants
    results["variable_params"] = all_variable

    for mm, match in enumerate(saved_exp_hashes):
        results[f"{match}"] = {}
        results[f"{match}"]["results"] = dat.load(
            save_location=f"results/{script_name}/{match}", parameters=result_keys
        )

        name_params = dat.load(save_location=f"params/{match}", parameters=all_variable)

        name = ""
        for kk, key in enumerate(all_variable):
            if kk > 0 and kk < len(all_variable):
                name += " | "
            name += f"{key}={name_params[key]}"

        results[f"{match}"]["name"] = name

    return results


def gen_parameter_variations(params, variation_dict):
    """
    Input the path to a json file containing the base parameters, and a dictionary of keys
    and a list of variations for the value.

    Returns a dictionary of all the variations possible and a unique hash created from the
    modified dictionaries.

    {
        experiment_hash1: dictionary of parameter variation,
        ...,
        experiment_hashn: dictionary of parameter variation,
    }
    """
    # for changing nested values
    def nested_set(dic, keys, value):
        for key in keys[:-1]:
            dic = dic.setdefault(key, {})
        dic[keys[-1]] = value

    variations = [
        dict(zip(variation_dict, v)) for v in product(*variation_dict.values())
    ]
    variation_dict = {}

    # do not overwrite the reference to params
    params_copy = copy.deepcopy(params)

    print(f"\nGeneratnig {len(variations)} variations of parameters\n")
    for vv, var in enumerate(variations):
        for key in var:
            nested_set(params_copy, key.split("/"), var[key])

        hash_name = gen_hash_name(params_copy)

        # print(f"Updated json_params by changing {key_list} to {var}\n{json_params}")
        print(f"hash_name: {hash_name}")
        print(params_copy)
        variation_dict[hash_name] = copy.deepcopy(params_copy)

    return variation_dict


def searchable_save(dat, results, params, script_name, overwrite=True):
    hash_name = gen_hash_name(params)

    # Save parameters
    dat.save(save_location=f"params/{hash_name}", data=params, overwrite=overwrite)

    dat.save(
        save_location=f"results/{script_name}/{hash_name}",
        data=results,
        overwrite=overwrite,
    )


def _example():
    import matplotlib.pyplot as plt

    # Instantiate database to save results
    db_name = "searchable_results_example"
    db_folder = None
    dat = DataHandler(db_name=db_name, database_dir=db_folder)

    # generate baseline json
    params = {
        "sin_params": {
            "A": 3,
            "shift": 5,
        },
        "time": [0, 5, 100],
        "exp": 2,
    }

    # if loading from json
    # with open(json_fp) as fp:
    #     params = json.load(fp)

    # example function that generates results
    # Needs to accept params dict as input and return dictionary of results
    def example_results(params):
        t = np.linspace(params["time"][0], params["time"][1], params["time"][2])
        y = (
            params["sin_params"]["A"]
            * np.sin(t - params["sin_params"]["shift"]) ** params["exp"]
        )
        return {"t": t, "y": y}

    # unique name for script that generates results
    # should update name if something changes in the script that would affect results
    script_name = "example_script"

    # get results
    print("--Getting results for baseline parameters--")
    results = example_results(params)

    # save in searchable format
    print("--Saving baseline results--")
    searchable_save(dat=dat, results=results, params=params, script_name=script_name)

    # helper function to quickly create some variations of our parameter set
    print("--Generating parameter variations--")
    param_variations = gen_parameter_variations(
        params=params, variation_dict={"sin_params/A": [5, 7, 10], "exp": [3, 4]}
    )

    # get results for each variation and save
    print("--Getting results for parameter variations--")
    for hash_id, varied_params in param_variations.items():
        print(f"\nGetting results for {hash_id}")
        # pretty printing of nested dictionaries
        print_nested(varied_params, indent=0, return_val=False)

        results = example_results(varied_params)
        print("Saving results")
        searchable_save(
            dat=dat, results=results, params=varied_params, script_name=script_name
        )

    # now load all results that have these parameter values
    const_params = {
        "exp": 3,
    }
    # result keys to load
    result_keys = ["y"]

    # Load results that have a set of common parameters
    print(f"Loading results with parameters:\n{const_params}")
    results = load_results(
        script_name=script_name,
        const_params=const_params,
        saved_exp_hashes=None,
        result_keys=result_keys,
        dat=dat,
        ignore_keys=None,
    )

    # plot the results
    plt.figure()
    ax = plt.subplot(111)
    for hash_name in results:
        # ignore const and variable params keys
        if "params" in hash_name:
            continue
        # print(dict_nested2str(results[hash_name]))
        ax.plot(results[hash_name]["results"]["y"], label=results[hash_name]["name"])

    # print the values that are constant between all tests
    ax.text(
        0,
        -5,
        (
            "Constant Parameters\n"
            + "___________________\n"
            + dict_nested2str(results["const_params"])
        ),
        fontsize=8,
    )
    plt.subplots_adjust(right=0.6)

    plt.legend()
    plt.tight_layout()
    plt.show()

    # Load results from a list of exp hashes
    # load_results(
    #     script_name=script_name,
    #     const_params=None,
    #     saved_exp_hashes=saved_exp_hashes,
    #     result_keys=result_keys,
    #     dat=dat,
    #     ignore_keys=None
    # )


if __name__ == "__main__":
    _example()
