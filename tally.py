#!/usr/bin/env python3
"""
Tally definitions for gamma-ray streaming simulation through concrete shield.
"""
import openmc
import numpy as np
from config import WALL_THICKNESS, SOURCE_TO_WALL_DISTANCE, MESH_DIMENSION, FINE_MESH_DIMENSION
from logging_utils import logger, LogSection

def create_tallies(cells, mesh_dimensions=None):
    """
    Create tallies for dose, flux, kerma, and heating.
    
    Parameters:
    -----------
    cells : dict
        Dictionary of simulation cells
    mesh_dimensions : list, optional
        Dimensions for mesh tallies [x, y, z]
    
    Returns:
    --------
    tallies : openmc.Tallies
        OpenMC tallies collection
    """
    with LogSection("Creating tallies"):
        tallies = openmc.Tallies()
        
        # Create a filter for the detector cell
        detector_filter = openmc.CellFilter(cells['detector'])
        
        # Create energy filter for energy-dependent tallies
        energy_filter = openmc.EnergyFilter(np.logspace(-3, 1, 100))  # 0.001 to 10 MeV
        
        # Create a filter for all cells (for mesh tallies)
        all_cells_filter = openmc.CellFilter(list(cells.values()))
        
        # 1. Flux tally in the detector
        flux_tally = openmc.Tally(name='Detector Flux')
        flux_tally.filters = [detector_filter, energy_filter]
        flux_tally.scores = ['flux']
        tallies.append(flux_tally)
        
        # 2. Heating tally in the detector (energy deposition)
        heating_tally = openmc.Tally(name='Detector Heating')
        heating_tally.filters = [detector_filter]
        heating_tally.scores = ['heating']
        tallies.append(heating_tally)
        
        # 3. Kerma tally in the detector
        kerma_tally = openmc.Tally(name='Detector Kerma')
        kerma_tally.filters = [detector_filter]
        kerma_tally.scores = ['kerma-photon']
        tallies.append(kerma_tally)
        
        # 4. Dose tally in the detector using ANS-6.1.1-1977
        dose_tally = openmc.Tally(name='Detector Dose')
        dose_tally.filters = [detector_filter, energy_filter]
        dose_tally.scores = ['flux']
        tallies.append(dose_tally)
        
        # Create mesh for spatial distribution
        if mesh_dimensions is None:
            mesh_dimensions = MESH_DIMENSION
            
        # Regular mesh spanning the problem space for visualization
        mesh = openmc.RegularMesh()
        mesh.dimension = mesh_dimensions
        mesh.lower_left = [-100, -100, -SOURCE_TO_WALL_DISTANCE-10]
        mesh.upper_right = [100, 100, WALL_THICKNESS + 200]
        
        # 5. Mesh tally for 2D visualization
        mesh_filter = openmc.MeshFilter(mesh)
        mesh_tally = openmc.Tally(name='Mesh Tally')
        mesh_tally.filters = [mesh_filter]
        mesh_tally.scores = ['flux']
        tallies.append(mesh_tally)
        
        # Fine mesh focused on region around channel exit
        fine_mesh = openmc.RegularMesh()
        fine_mesh.dimension = FINE_MESH_DIMENSION if mesh_dimensions is None else [d*2 for d in mesh_dimensions]
        fine_mesh.lower_left = [-20, -20, WALL_THICKNESS - 5]
        fine_mesh.upper_right = [20, 20, WALL_THICKNESS + 50]
        
        # 6. Fine mesh tally for detailed visualization around channel exit
        fine_mesh_filter = openmc.MeshFilter(fine_mesh)
        fine_mesh_tally = openmc.Tally(name='Fine Mesh Tally')
        fine_mesh_tally.filters = [fine_mesh_filter]
        fine_mesh_tally.scores = ['flux']
        tallies.append(fine_mesh_tally)
        
        logger.info(f"Created {len(tallies)} tallies for flux, heating, kerma, dose, and mesh visualization")
        
        return tallies
