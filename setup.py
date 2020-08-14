import os
import setuptools
from setuptools import find_packages, setup

NAME = 'heinsight_liquid_level_H1'
DESCRIPTION = """HeinSight is a computer-vision based control system for automating common laboratory tasks. The Liquid-Level Hydrogen-1 release is a prototype designed to facilitate the remote monitoring and control of liquid-air interface in a transparent vessel across a variety of pre-programmed experimental types that require continuous stirring: single and dual pump continuous preferential crystallization (CPC), continuous distillation, and filtration"""
AUTHOR = 'Veronica Lai, Tara Zepel, Lars Yunker // Hein Group'
CLASSIFIERS = [
    'Programming Language :: Python :: 3',
    'Operating System :: Windows',
]
INSTALL_REQUIRES = [
    'opencv',
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
# PACKAGES = [NAME] + [f'{NAME}.{package}' for package in find_packages(exclude=EXCLUDE_PACKAGES)]
PACKAGES = find_packages()

setup(
    name=NAME,
    version='1.0',
    description=DESCRIPTION,
    author=AUTHOR,
    url='https://gitlab.com/heingroup/heinsight_liquid_level_H1',
    install_requires=INSTALL_REQUIRES,
    packages=PACKAGES,
    license=lic,
    classifiers=CLASSIFIERS,
)
