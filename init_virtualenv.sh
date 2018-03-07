#!/bin/bash

# Initialize virtual Python environment
virtualenv py2_virtualenv

# This activates the environment.
source py2_virtualenv/bin/activate

# Install packages
pip install --upgrade pip
pip install tensorflow
pip install keras
pip install h5py
pip install pypng
pip install sklearn
pip install matplotlib
pip install numpy
pip install jupyter
