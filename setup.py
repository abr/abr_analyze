import io
from setuptools import setup, find_packages

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

setup_requires = ["setuptools>=18.0", "numpy", "seaborn==0.7.1", "h5py==2.8.0",
        "Pillow==5.1.0", "terminaltables==3.1.0", "redis==2.10.5", "pytest"]

setup(
    name='abr_analyze',
    version='0.1',
    description='Tools for analyzing data',
    url='https://github.com/abr/abr_analyze',
    author='Applied Brain Research',
    author_email='pawel.jaworski@appliedbrainresearch.com',
    license="Free for non-commercial use",
    long_description=read('README.md'),
    install_requires=setup_requires + [
        "matplotlib", "scipy",
    ],
    packages=find_packages(),
)
