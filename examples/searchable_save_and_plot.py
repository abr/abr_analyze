import numpy as np

from abr_analyze import DataHandler, data_logger

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
data_logger.searchable_save(dat=dat, results=results, params=params, script_name=script_name)

# helper function to quickly create some variations of our parameter set
print("--Generating parameter variations--")
param_variations = data_logger.gen_parameter_variations(
    params=params, variation_dict={"sin_params/A": [5, 7, 10], "exp": [3, 4]}
)

# get results for each variation and save
print("--Getting results for parameter variations--")
for hash_id, varied_params in param_variations.items():
    print(f"\nGetting results for {hash_id}")
    # pretty printing of nested dictionaries
    data_logger.print_nested(varied_params, indent=0, return_val=False)

    results = example_results(varied_params)
    print("Saving results")
    data_logger.searchable_save(
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
results = data_logger.load_results(
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
        + data_logger.dict_nested2str(results["const_params"])
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

