import os

from urllib.request import urlretrieve

from abr_analyze.paths import database_dir

def check_exists():
    if not os.path.isfile('%s/abr_analyze_examples.h5'%database_dir):
        print('First time running...Downloading examples database')
        db_loc = ("https://drive.google.com/uc?export=download&"
            + "id=1bojKaBsXHrMe9NgvaJiUx3pHOSglY0na")
        urlretrieve(
            url=db_loc,
            filename='%s/abr_analyze_examples.h5'%database_dir)
        print('Examples database saved to %s'%database_dir)
