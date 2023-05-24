#!/usr/bin/env bash

#making virtual environment
python3 -m venv assignment4_viz_env

source ./assignment4_viz_env/bin/activate

#upgrading pip and installing requirements
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

