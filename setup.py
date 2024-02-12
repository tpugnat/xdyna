# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from setuptools import setup, find_packages, Extension
from pathlib import Path

#######################################
# Prepare list of compiled extensions #
#######################################

extensions = []

#########
# Setup #
#########

version_file = Path(__file__).parent / 'xdyna/_version.py'
dd = {}
with open(version_file.absolute(), 'r') as fp:
    exec(fp.read(), dd)
__version__ = dd['__version__']

setup(
    name='xdyna',
    version=__version__,
    description='Dynamic aperture library for particle accelerators',
    long_description='Dynamic aperture library for particle accelerators',
    url='https://xsuite.readthedocs.io/',
    author='G. Iadarola et al.',
    license='Apache 2.0',
    project_urls={
            "Bug Tracker": "https://github.com/xsuite/xdyna/issues",
            "Documentation": 'https://xsuite.readthedocs.io/',
            "Source Code": "https://github.com/xsuite/xdyna",
        },
    packages=find_packages(),
    ext_modules = extensions,
    include_package_data=True,
    install_requires=[
        'numpy>=1.0',
        "pandas>=2.0",
        'scipy',
#         'json',
        'xobjects',
        'xdeps',
        'xpart',
        'xtrack',
        'cpymad',
        ],
    extras_require={
        'tests': [ 'NAFFlib', 'PyHEADTAIL', 'pytest', 'pytest-mock'],
        },
    )
