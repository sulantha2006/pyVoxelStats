from setuptools import setup, find_packages

setup(
    # Application name:
    name="pyVoxelStats",

    # Version number (initial):
    version="0.1.1a15",

    # Application author details:
    author="Sulantha Mathotaarachchi",
    author_email="sulantha.ms@gmail.com",

    # Packages
    packages=find_packages(),

    package_data = {'pyVS': ['data/*']},

    # Include additional files into the package
    include_package_data=True,

    # Details
    url="http://pypi.python.org/pypi/pyVoxelStats_v011a15/",

    #
    # license="LICENSE.txt",
    description="Python implementation of the VoxelStats toolbox for neuroimage analysis",

    # long_description=open("README.txt").read(),

    # Dependent packages (distributions)
    install_requires=[
        "nibabel", "pandas", "numpy",
        "pyminc",
        "statsmodels",
        "rpy2",
        "ipyparallel",
    ],
)