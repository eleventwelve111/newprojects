#!/usr/bin/env python3
"""
Configuration settings for gamma-ray streaming simulation through concrete shield.
"""
import numpy as np
import os
import pathlib
import json
import logging
from typing import Dict, Any, Optional

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

# Mesh parameters for tallies - adding this missing constant
MESH_DIMENSION = [30, 30, 40]  # Default mesh dimensions
FINE_MESH_DIMENSION = [60, 60, 80]  # Fine mesh for higher resolution

# Simulation parameters
DEFAULT_PARTICLES = 1000000  # Default number of particles per simulation
DEFAULT_BATCHES = 10  # Default number of batches
DEFAULT_INACTIVE = 5  # Default number of inactive batches

class Config:
    """Class to handle configuration settings with save/load functionality."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration settings, optionally from a file."""
        # Set default values from module constants
        self.wall_thickness = WALL_THICKNESS
        self.source_to_wall_distance = SOURCE_TO_WALL_DISTANCE
        self.channel_diameters = CHANNEL_DIAMETERS.tolist()
        self.source_energies = SOURCE_ENERGIES.tolist()
        self.detector_diameter = DETECTOR_DIAMETER
        self.detector_distances = DETECTOR_DISTANCES.tolist()
        self.detector_angles = DETECTOR_ANGLES.tolist()
        self.mesh_dimension = MESH_DIMENSION
        self.fine_mesh_dimension = FINE_MESH_DIMENSION
        self.default_particles = DEFAULT_PARTICLES
        self.default_batches = DEFAULT_BATCHES
        self.default_inactive = DEFAULT_INACTIVE
        
        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'wall_thickness': self.wall_thickness,
            'source_to_wall_distance': self.source_to_wall_distance,
            'channel_diameters': self.channel_diameters,
            'source_energies': self.source_energies,
            'detector_diameter': self.detector_diameter,
            'detector_distances': self.detector_distances,
            'detector_angles': self.detector_angles,
            'mesh_dimension': self.mesh_dimension,
            'fine_mesh_dimension': self.fine_mesh_dimension,
            'default_particles': self.default_particles,
            'default_batches': self.default_batches,
            'default_inactive': self.default_inactive
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def load_from_file(self, filepath: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            # Update attributes from loaded data
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.error(f"Error loading configuration from {filepath}: {str(e)}")
    
    def get_parameter_space(self) -> Dict[str, list]:
        """Get the parameter space for parametric studies."""
        return {
            'energy': self.source_energies,
            'channel_diameter': self.channel_diameters,
            'detector_distance': self.detector_distances,
            'detector_angle': self.detector_angles
        }

# Create a default configuration instance
default_config = Config()
