import os
import sys

# Set the path based on the operating system

if sys.platform.startswith('win'):  # pylint: disable=R1720
    raise Exception('Currently not supported for Windows')

else:
    # the repo root directory
    dir_name = os.path.dirname(__file__)
    home_dir = os.path.expanduser('~')
    current_dir = os.path.abspath('.')

    # create file for custom paths if it does not exist
    with open("%s/paths.txt" % dir_name, "a+") as f:
        pass

    with open("%s/paths.txt" % dir_name, "r+") as f:
        paths = f.read().split('\n')

        cache_dir = os.path.join(current_dir, "data/.cache")
        database_dir = os.path.join(current_dir, "data/databases")
        figures_dir = os.path.join(current_dir, "data/figures")

        # check if the template has been setup by checking is substrings are present
        if not any('cache_dir' in x for x in paths):
            f.write('cache_dir:\n')
            f.truncate()

        if not any('database_dir' in x for x in paths):
            f.write('database_dir:\n')
            f.truncate()

        if not any('figures_dir' in x for x in paths):
            f.write('figures_dir:\n')
            f.truncate()

        # check if any custom paths have been setup
        for line in paths:
            if 'cache_dir' in line:
                path = line.split(':')[1]
                if len(path) > 0:
                    cache_dir = path
            if 'database_dir' in line:
                path = line.split(':')[1]
                if len(path) > 0:
                    database_dir = path
            if 'figures_dir' in line:
                path = line.split(':')[1]
                if len(path) > 0:
                    figures_dir = path

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(database_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

if __name__ == '__main__':
    print('===== Current Save Directories =====')
    print('cache_dir: ', cache_dir)
    print('database_dir: ', database_dir)
    print('figures_dir: ', figures_dir)
    print('====================================')
    print('\nNOTE: to change your locations, add the direct path to'
          + ' abr_analyze/abr_analyze/paths.txt\n'
          + 'Note that the default locations will be a "data" folder in your'
          + ' current working directory.')
