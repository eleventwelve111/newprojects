#!/usr/bin/env python3
"""
Tally definitions for gamma-ray streaming simulation through concrete shield.
"""
import openmc
import numpy as np
from config import MESH_DIMENSION, FINE_MESH_DIMENSION, WALL_THICKNESS, SOURCE_TO_WALL_DISTANCE
from logging_utils import logger, LogSection

def create_tallies(channel_diameter):
    """
    Create tallies for dose, flux, kerma, and heating.
    
    Parameters:
    -----------
    channel_diameter : float
        Diameter of the air channel in cm
    
    Returns:
    --------
    tallies : openmc.Tallies
        OpenMC tallies collection
    """
    with LogSection(f"Creating tallies (channel Ø={channel_diameter} cm)"):
        tallies = openmc.Tallies()
        
        # Create a filter for the detector cell
        cell_filter = openmc.CellFilter([3])  # Assuming detector is cell 3
        
        # Create energy filter for energy-dependent tallies
        energy_filter = openmc.EnergyFilter(np.logspace(-3, 1, 100))  # 0.001 to 10 MeV
        
        # Define dose conversion coefficients based on NCRP-38, ANS-6.1.1-1977
        # Energy (MeV): [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0, 1.4, 1.8, 2.2, 2.6, 2.8, 3.25, 3.75, 4.25, 4.75, 5.0, 5.25, 5.75, 6.25, 6.75, 7.5, 9.0, 11.0, 13.0, 15.0]
        energy_points = np.array([
            0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 
            0.65, 0.7, 0.8, 1.0, 1.4, 1.8, 2.2, 2.6, 2.8, 3.25, 3.75, 4.25, 4.75, 5.0, 5.25, 
            5.75, 6.25, 6.75, 7.5, 9.0, 11.0, 13.0, 15.0
        ])
        
        # Flux-to-dose conversion factors ((rem/hr)/(photons/cm²-s))
        dose_coeffs = np.array([
            3.96e-6, 5.82e-7, 2.90e-7, 2.58e-7, 2.83e-7, 3.79e-7, 5.01e-7, 6.31e-7, 7.59e-7, 
            8.78e-7, 9.85e-7, 1.08e-6, 1.17e-6, 1.27e-6, 1.36e-6, 1.44e-6, 1.52e-6, 1.68e-6, 
            1.98e-6, 2.51e-6, 2.99e-6, 3.42e-6, 3.82e-6, 4.01e-6, 4.41e-6, 4.83e-6, 5.23e-6, 
            5.60e-6, 5.80e-6, 6.01e-6, 6.37e-6, 6.74e-6, 7.11e-6, 7.66e-6, 8.77e-6, 1.03e-5, 
            1.18e-5, 1.33e-5
        ])
        
        # Create function filter for dose conversion
        dose_function = openmc.FunctionFilter(lambda x: np.interp(x, energy_points, dose_coeffs))
        
        # Create a photon flux tally
        flux_tally = openmc.Tally(name='detector_flux')
        flux_tally.filters = [cell_filter]
        flux_tally.scores = ['flux']
        tallies.append(flux_tally)
        
        # Create a photon energy-dependent flux tally
        energy_flux_tally = openmc.Tally(name='detector_flux_spectrum')
        energy_flux_tally.filters = [cell_filter, energy_filter]
        energy_flux_tally.scores = ['flux']
        tallies.append(energy_flux_tally)
        
        # Create a dose tally using flux-to-dose conversion
        dose_tally = openmc.Tally(name='detector_dose')
        dose_tally.filters = [cell_filter, dose_function]
        dose_tally.scores = ['flux']
        tallies.append(dose_tally)
        
        # Create a heating tally
        heating_tally = openmc.Tally(name='detector_heating')
        heating_tally.filters = [cell_filter]
        heating_tally.scores = ['heating']
        tallies.append(heating_tally)
        
        # Create a kerma tally
        kerma_tally = openmc.Tally(name='detector_kerma')
        kerma_tally.filters = [cell_filter]
        kerma_tally.scores = ['kerma']
        tallies.append(kerma_tally)
        
        # Create a regular mesh tally for dose distribution
        mesh = openmc.RegularMesh()
        mesh.dimension = MESH_DIMENSION
        mesh.lower_left = (-100, -100, WALL_THICKNESS)
        mesh.upper_right = (100, 100, WALL_THICKNESS + 200)
        
        mesh_filter = openmc.MeshFilter(mesh)
        mesh_tally = openmc.Tally(name='dose_mesh')
        mesh_tally.filters = [mesh_filter, dose_function]
        mesh_tally.scores = ['flux']
        tallies.append(mesh_tally)
        
        # Create a fine mesh tally near the channel exit for detailed analysis
        fine_mesh = openmc.RegularMesh()
        fine_mesh.dimension = FINE_MESH_DIMENSION
        fine_mesh.lower_left = (-20, -20, WALL_THICKNESS - 1)
        fine_mesh.upper_right = (20, 20, WALL_THICKNESS + 40)
        
        fine_mesh_filter = openmc.MeshFilter(fine_mesh)
        fine_mesh_tally = openmc.Tally(name='Fine Mesh Dose Distribution')
        fine_mesh_tally.filters = [fine_mesh_filter, dose_function]
        fine_mesh_tally.scores = ['flux']
        tallies.append(fine_mesh_tally)
        
        # Create a mesh for source visualization
        source_mesh = openmc.RegularMesh()
        source_mesh.dimension = [50, 50, 1]
        source_mesh.lower_left = (-25, -25, -SOURCE_TO_WALL_DISTANCE - 10)
        source_mesh.upper_right = (25, 25, -SOURCE_TO_WALL_DISTANCE + 10)
        
        source_mesh_filter = openmc.MeshFilter(source_mesh)
        source_mesh_tally = openmc.Tally(name='Source Visualization')
        source_mesh_tally.filters = [source_mesh_filter]
        source_mesh_tally.scores = ['flux']
        tallies.append(source_mesh_tally)
        
        # Create a mesh for channel visualization
        channel_mesh = openmc.RegularMesh()
        channel_mesh.dimension = [50, 50, 10]
        channel_mesh.lower_left = (-channel_diameter*2, -channel_diameter*2, -1)
        channel_mesh.upper_right = (channel_diameter*2, channel_diameter*2, WALL_THICKNESS + 1)
        
        channel_mesh_filter = openmc.MeshFilter(channel_mesh)
        channel_mesh_tally = openmc.Tally(name='Channel Visualization')
        channel_mesh_tally.filters = [channel_mesh_filter]
        channel_mesh_tally.scores = ['flux']
        tallies.append(channel_mesh_tally)
        
        # Create a track-length tally for radiography-style visualization
        track_mesh = openmc.RegularMesh()
        track_mesh.dimension = [200, 200, 1]
        track_mesh.lower_left = (-50, -50, WALL_THICKNESS + 0.1)
        track_mesh.upper_right = (50, 50, WALL_THICKNESS + 0.2)
        
        track_mesh_filter = openmc.MeshFilter(track_mesh)
        track_tally = openmc.Tally(name='Radiography View')
        track_tally.filters = [track_mesh_filter]
        track_tally.scores = ['flux']
        tallies.append(track_tally)
        
        logger.info(f"Created {len(tallies)} tallies for dose, flux, kerma, heating, and mesh distributions")
        
        # Export to XML for visualization and verification
        tallies.export_to_xml()
        
        return tallies
