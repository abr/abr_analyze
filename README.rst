***********
ABR Analyze
***********

ABR_Analyze: A repository for saving, processing, and plotting data from an hdf5 database.

Installation
============

The ABR_Analyze repo depends on NumPy, SciPy, Matplotlib, and H5Py, and we recommend that
you install these libraries before ABR_Analyze. If you're not sure how to do this,
we recommend using `Anaconda <https://store.continuum.io/cshop/anaconda/>`_.
Note that installing in a clean environment will require compiling of the
dependent libraries, and will take a few minutes.

ABR_Analyze is tested to work on Python 3.4+.


Setting paths
=============

See abr_analyze/utils/paths.py and adjust your default locations for
databases, figures, and caches items

A default paths.py is committed, to avoid running into conflicts with
you personal setup, run::

   git update-index --assume-unchanged path_to_paths.py

this will assume the file is unchanged and will not commit it.

Usage
=====

TODO
