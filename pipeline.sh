#!/bin/bash

# Call script with: source pipeline.sh in the directory

# Runs the scripts
conda activate tsht

# # --- (0) Preliminary: generate RWD --- #
# python3 0_datasets.py

# # --- (1) Run the simulations --- #
# python3 -W ignore::RuntimeWarning 1_simulations.py

# # --- (2) Generate additional figures --- #
# python3 2_extra_figs.py

# --- (3) Run the regression simulation (fake data) --- #
python3 3_regression_sim.py

# # --- (4) Run the regression simulation (real data) --- #
# python3 4_regression_rwd.py



echo "~~~ End of pipeline.sh ~~~"
