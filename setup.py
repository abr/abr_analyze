import io
import runpy
import os
from setuptools import setup, find_packages

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

root = os.path.dirname(os.path.realpath(__file__))
version = runpy.run_path(
    os.path.join(root, 'abr_analyze', 'version.py'))['version']

setup_requires = [
    "setuptools>=18.0",
    ]

install_requires = [
    "h5py==2.8.0",
    "Pillow==5.1.0",
    "terminaltables==3.1.0",
    "redis==2.10.5",
    "numpy>=1.13.3",
    "matplotlib>=3.0.2",
    "scipy==1.1.0",
    "nengo>=2.8.0",
    "nengolib>=0.5.2",
    "nengo_extras>=0.3.0"
    ]

tests_require = [
    "pytest>=4.3.0",
    "pytest-xdist>=1.26.0",
    "pytest-cov>=2.6.0",
    "coverage>=4.5.0"
    "pytest-plt"
    ]


setup(
    name='abr_analyze',
    version=version,
    description='Tools for analyzing data',
    url='https://github.com/abr/abr_analyze',
    author='Applied Brain Research',
    author_email='pawel.jaworski@appliedbrainresearch.com',
    license="Free for non-commercial use",
    long_description=read('README.rst'),
    install_requires=install_requires + setup_requires,
    setup_requires=setup_requires,
    extras_require={"tests": tests_require},
    packages=find_packages(),
)
