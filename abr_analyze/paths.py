import os
import sys

# Set the path based on the operating system

if sys.platform.startswith('win'):  # pylint: disable=R1720
    raise Exception('Currently not supported for Windows')
    # config_dir = os.path.expanduser(os.path.join("~", ".abr_analyze"))
    # cache_dir = os.path.join(config_dir, "cache")
    # database_dir = os.path.joint(cache_dir, "abr_analyze_db.h5")
else:
    home_dir = os.path.expanduser('~')
    current_dir = os.path.abspath('.')

    cache_dir = os.path.abspath(
        os.path.join(home_dir, "Data/.cache", "abr_analyze"))
    # the repo root directory
    dir_name = os.path.dirname(__file__)
    #database_dir = os.path.abspath(os.path.join(dir_name, "..", "databases"))
    database_dir = os.path.abspath(os.path.join(home_dir, "Data/abr_analyze", "databases"))
    figures_dir = os.path.join(current_dir, "Figures")

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(database_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs('%s'%figures_dir, exist_ok=True)
