import os

from . import nengo, plotting, utils
from .data_handler import DataHandler
from .version import version as __version__

# from . import gui


dir_name = os.path.dirname(__file__)
if not os.path.isfile:
    with open("%s/paths.txt" % dir_name, "a+") as f:
        f.write("cache_dir:\n")
        f.write("database_dir:\n")
        f.write("figures_dir:\n")
