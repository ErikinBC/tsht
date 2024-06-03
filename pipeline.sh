#!/bin/bash

# Call script with: source pipeline.sh in the directory

# Runs the scripts
conda activate tsht

# --- (1) Run the simulations --- #
python3 -W ignore::RuntimeWarning 1_simulations.py

# --- (2) Generate additional figures --- #
python3 2_extra_figs.py

# --- (3) Run the regression simulations --- #
python3 3_regression.py



echo "~~~ End of pipeline.sh ~~~"
