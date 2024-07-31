# Simulation and Plotting Instructions

## Overview

This document provides instructions for running the BottleCOEX simulator and generating plots from its output. The simulation reads configuration files to set up and run experiments, and the results are saved and visualized using subsequent scripts.

## Step-by-Step Guide

### 1. Run BottleCOEX.py

The `BottleCOEX.py` script is the main simulator. It reads the path specified in `Config.json` and loads the following files from that path:
- `EData.json`
- `paramsData.json`
- Network description files (by default, only the two-route bottleneck `Net2Plain` is loaded. Sample other networks are in the `./Networks` folder).

#### Default Path
- Path `0` is the current folder and contains a sample experiment that runs quickly and produces output promptly.
- To use data from a different experiment, set the path number to a different value (to reproduce the experiments stored in the `./Experiments` folder).
- To create your own experiment, modify the `EData.json` and/or `paramsData.json` files and/or add a new path where the data for the experiment are contained. Allowed parameter values are specified in the file `SimulatorParamNames.txt`.

### 2. Output Generation

The output of `BottleCOEX.py` is saved into the path folder in a subfolder named based on the experiment name and the varied parameters. For example, for the following values from `EData.json`:

### 3. Run BottlePlotterTimeDep.py

The BottlePlotterTimeDep.py script generates plots from the experiment results using the output `.csv` files generated in the previous step. This script creates a grid of time-dependent plots of travel times and vehicle counts. These plots correspond to the supplementary figures in the manuscript submitted to xxx.

### 4. Run TODOTODO

This plotter script generates the summary figures used in the manuscript submitted to xxx.
