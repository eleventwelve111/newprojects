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
REPORT_DIR = RESULTS_DIR / "report"

for directory in [RESULTS_DIR, PLOTS_DIR, DATA_DIR, REPORT_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Physical dimensions
INCH_TO_CM = 2.54
FOOT_TO_CM = 12 * INCH_TO_CM

# Wall parameters
WALL_THICKNESS = 2 * FOOT_TO_CM  # 2 ft in cm
SOURCE_TO_WALL_DISTANCE = 6 * FOOT_TO_CM  # 6 ft in cm

# Channel parameters
CHANNEL_DIAMETERS = np.array([0.05, 0.1, 0.5, 1.0])  # cm

# Source parameters
SOURCE_ENERGIES = np.array([0.1, 0.5, 1.0, 2.0, 5.0])  # MeV

# Detector parameters
DETECTOR_DIAMETER = 30.0  # cm (ICRU sphere)
DETECTOR_DISTANCES = np.array([30, 40, 60, 80, 100, 150])  # cm
DETECTOR_ANGLES = np.array([0, 5, 10, 15, 30, 45])  # degrees

# Simulation parameters
PARTICLES = int(1e6)  # default number of particles
BATCHES = 100
INACTIVE_BATCHES = 10
RANDOM_SEED = 42

# Mesh parameters
MESH_DIMENSION = [100, 100, 1]  # Regular mesh size
FINE_MESH_DIMENSION = [200, 200, 1]  # Fine mesh size

# Concrete types for comparison
CONCRETE_TYPES = ["standard", "barite", "magnetite"]

# Safety limits (rem/year)
OCCUPATIONAL_LIMIT = 5.0  # rem/year
PUBLIC_LIMIT = 0.1  # rem/year

# Convert to rem/hr assuming 2000 work hours/year for occupational and 8760 hours/year for public
OCCUPATIONAL_LIMIT_HR = OCCUPATIONAL_LIMIT / 2000
PUBLIC_LIMIT_HR = PUBLIC_LIMIT / 8760

# File names
RESULTS_JSON = DATA_DIR / "simulation_results.json"
CHECKPOINT_FILE = DATA_DIR / "checkpoint.json"
REPORT_FILE = REPORT_DIR / "analysis_report.md"

# Machine learning parameters
ML_FEATURES = ['energy', 'channel_diameter', 'distance', 'angle']
ML_TARGET = 'dose'
ML_TEST_SIZE = 0.2
ML_RANDOM_STATE = 42
