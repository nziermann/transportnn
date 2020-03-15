from setuptools import find_packages
from setuptools import setup
import subprocess

REQUIRED_PACKAGES = ['netCDF4', 'talos', 'keras', 'numpy']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)
