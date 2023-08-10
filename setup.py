import os
import setuptools
from setuptools import find_packages, setup

NAME = 'gradient_liquid_level'
DESCRIPTION = """This module is designed to identify the liquid level 
    in a vessel using computer vision techniques.  The main algorithm used is 
    inspired by the work in this paper:  [paper link]
    The structure of the package is largely taken from the HeinSeight package
    developed by Veronica Lai, Tara Zepel, Lars Yunker of the Hein Group:
    [paper link]
    """
AUTHOR = 'Emma Brass // Cooper Group'
CLASSIFIERS = [
    'Programming Language :: Python :: 3',
    'Operating System :: Windows',
]
INSTALL_REQUIRES = [ # TODO check the requirements are right
    'opencv-python',
    'imutils',
    'numpy',
    'pandas',
    'matplotlib',
    'pyqt5',
    'pyqt5-tools',
    'slackclient==2.0.1',
    'pillow',
    'pyserial'
]

with open('LICENSE') as f:
    lic = f.read()
    lic.replace('\n', ' ')

# find packages and prefix them with the main package name
# PACKAGES = [NAME] + [f'{NAME}.{package}' 
# for package in find_packages(exclude=EXCLUDE_PACKAGES)]
PACKAGES = find_packages()

setup(
    name=NAME,
    version='1.0',
    description=DESCRIPTION,
    author=AUTHOR,
    url='https://gitlab.com/heingroup/heinsight_liquid_level_H1', #TODO change
    install_requires=INSTALL_REQUIRES,
    packages=PACKAGES,
    license=lic,
    classifiers=CLASSIFIERS,
)
