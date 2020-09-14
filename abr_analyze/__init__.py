from .version import version as __version__

from . import utils
from . import plotting
from . import nengo_utils
from . import gui

from .data_handler import DataHandler

import os

dir_name = os.path.dirname(__file__)
if not os.path.isfile:
    with open("%s/paths.txt" % dir_name, "a+") as f:
        f.write('cache_dir:\n')
        f.write('database_dir:\n')
        f.write('figures_dir:\n')
