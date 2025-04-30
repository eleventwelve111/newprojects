#!/usr/bin/env python3
"""
Configuration settings for gamma-ray streaming simulation through concrete shield.
"""
import numpy as np
import os
import pathlib

# Create results directory if it doesn't exist
RESULTS_DIR = pathlib.Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
DATA_DIR = RESULTS_DIR / "data"
REPORT_DIR = RESULTS_DIR / "reports"
MODEL_DIR = RESULTS_DIR / "models"

for directory in [RESULTS_DIR, PLOTS_DIR, DATA_DIR, REPORT_DIR, MODEL_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Physical dimensions (convert from imperial to metric)
INCH_TO_CM = 2.54
FOOT_TO_CM = 12 * INCH_TO_CM

# Wall parameters
WALL_THICKNESS = 2 * FOOT_TO_CM  # 2 ft in cm
SOURCE_TO_WALL_DISTANCE = 6 * FOOT_TO_CM  # 6 ft in cm

# Channel parameters (mm to cm conversion for smallest diameter)
CHANNEL_DIAMETERS = np.array([0.05, 0.1, 0.5, 1.0])  # cm

# Source parameters
SOURCE_ENERGIES = np.array([0.1, 0.5, 1.0, 2.0, 5.0])  # MeV

# Detector parameters
DETECTOR_DIAMETER = 30.0  # cm (ICRU sphere)
DETECTOR_DISTANCES = np.array([30, 40, 60, 80, 100, 150])  # cm from back of wall
DETECTOR_ANGLES = np.array([0, 5, 10, 15, 30, 45])  # degrees off-axis

# Mesh settings for visualization
MESH_DIMENSION = [100, 100, 100]  # General mesh
FINE_MESH_DIMENSION = [200, 200, 200]  # Finer mesh for detailed analysis

# Simulation parameters
PARTICLES_PER_BATCH = 10000
NUM_BATCHES = 100
NUM_INACTIVE_BATCHES = 10

# Checkpoint frequency (batches)
CHECKPOINT_FREQUENCY = 10

# Random seed for reproducibility
RANDOM_SEED = 42
