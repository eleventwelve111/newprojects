#!/usr/bin/env python3
"""
Weight window generation and application for variance reduction in OpenMC.
"""
import numpy as np
import openmc
import matplotlib.pyplot as plt
from logging_utils import logger, LogSection, timeit
from config import MESH_DIMENSION

@timeit
def generate_weight_windows(model, source, particles=1e5):
    """
    Generate weight windows based on an initial low-statistics run.
    
    Parameters:
    -----------
    model : dict
        Dictionary containing model components
    source : openmc.Source
        Source definition
    particles : int
        Number of particles for weight window generation
    
    Returns:
    --------
    weight_windows : dict
        Weight window parameters
    """
    with LogSection("Generating weight windows"):
        # Extract model components
        materials = model['materials']
        geometry = model['geometry']
        cells = model['cells']
        run_dir = model['run_dir']
        
        # Create a mesh for weight windows
        ww_mesh = openmc.RegularMesh()
        ww_mesh.dimension = MESH_DIMENSION
        
        # Set mesh boundaries to cover the entire problem
        ww_mesh.lower_left = [-50, -50, -100]
        ww_mesh.upper_right = [50, 50, 200]
        
        # Create mesh filter and tally for weight windows
        mesh_filter = openmc.MeshFilter(ww_mesh)
        importance_tally = openmc.Tally(name='importance')
        importance_tally.filters = [mesh_filter]
        importance_tally.scores = ['flux']
        
        # Create a new tallies object for the weight window generation run
        tallies = openmc.Tallies([importance_tally])
        
        # Create settings for weight window generation run
        settings = openmc.Settings()
        settings.run_mode = 'fixed source'
        settings.particles = int(particles)
        settings.batches = 10
        settings.source = source
        
        # Export modified XML files for weight window generation
        ww_dir = run_dir / "weight_windows"
        ww_dir.mkdir(exist_ok=True)
        
        materials.export_to_xml(ww_dir / "materials.xml")
        geometry.export_to_xml(ww_dir / "geometry.xml")
        settings.export_to_xml(ww_dir / "settings.xml")
        tallies.export_to_xml(ww_dir / "tallies.xml")
        
        # Run OpenMC for weight window generation
        logger.info(f"Running OpenMC for weight window generation ({settings.particles} particles)")
        openmc.run(cwd=str(ww_dir), output=False)
        
        # Load statepoint and extract mesh tally data
        sp_file = list(ww_dir.glob("statepoint.*.h5"))[0]
        sp = openmc.StatePoint(sp_file)
        
        # Get the flux tally
                # Get the flux tally
        flux_tally = sp.get_tally(name='importance')
        
        # Extract the mesh tally data
        mesh_data = flux_tally.get_values().reshape(MESH_DIMENSION)
        
        # Generate weight windows based on the mesh tally data
        # The weight window lower bounds are typically set to 1/5 of the cell importance
        
        # Avoid division by zero by replacing zeros with minimum non-zero value
        min_nonzero = np.min(mesh_data[mesh_data > 0]) if np.any(mesh_data > 0) else 1.0
        mesh_data = np.maximum(mesh_data, min_nonzero * 1e-10)
        
        # Calculate ratios between adjacent cells for gradient-based weight windows
        # We'll use a simple approach here - calculate importance as flux relative to average
        avg_importance = np.mean(mesh_data)
        importance_values = mesh_data / avg_importance
        
        # Set weight window lower bounds
        lower_bounds = importance_values / 5.0
        
        # Set weight window upper bounds (typically 5x the lower bound)
        upper_bounds = lower_bounds * 5.0
        
        # Create weight_windows object for OpenMC
        weight_windows = {
            'mesh': ww_mesh,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'importance_values': importance_values
        }
        
        # Save weight windows data for later use
        np.savez(ww_dir / "weight_windows.npz", 
                 lower_bounds=lower_bounds,
                 upper_bounds=upper_bounds,
                 importance_values=importance_values)
        
        # Create a visualization of the weight windows
        plot_weight_windows(ww_mesh, importance_values, run_dir / "weight_windows_importance.png")
        
        logger.info(f"Weight windows generated based on {particles} particle run")
        
        return weight_windows

@timeit
def apply_weight_windows(model, weight_windows):
    """
    Apply weight windows to the model for variance reduction.
    
    Parameters:
    -----------
    model : dict
        Dictionary containing model components
    weight_windows : dict
        Weight window parameters
    
    Returns:
    --------
    settings : openmc.Settings
        Updated OpenMC settings with weight windows
    """
    with LogSection("Applying weight windows"):
        # Extract components
        settings = model['settings']
        run_dir = model['run_dir']
        
        # Extract weight window data
        ww_mesh = weight_windows['mesh']
        lower_bounds = weight_windows['lower_bounds']
        upper_bounds = weight_windows['upper_bounds']
        
        # Create weight windows file
        ww_file = run_dir / "weight_windows.xml"
        
        # Apply the weight windows to the settings
        settings.weight_windows = {
            'mesh': ww_mesh,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds
        }
        
        # Update survival_biasing
        settings.survival_biasing = True
        
        # Export updated settings
        settings.export_to_xml(run_dir / "settings.xml")
        
        logger.info(f"Weight windows applied to model")
        
        return settings

def plot_weight_windows(mesh, importance_values, output_file):
    """
    Plot weight window importance values.
    
    Parameters:
    -----------
    mesh : openmc.RegularMesh
        Mesh used for weight windows
    importance_values : numpy.ndarray
        Importance values for each mesh cell
    output_file : str or Path
        Output file path
    """
    # Create a slice through the center of the mesh
    central_slice = importance_values[:, mesh.dimension[1]//2, :]
    
    # Take log of importance for better visualization
    log_importance = np.log10(central_slice)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot importance map
    im = ax.imshow(log_importance.T, origin='lower', aspect='auto', cmap='jet',
                   extent=[-50, 50, -100, 200])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Log10(Importance)', rotation=270, labelpad=20)
    
    # Add labels and title
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Z (cm)')
    ax.set_title('Weight Window Importance Values (Y-center slice)')
    
    # Add annotation for key areas
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
    ax.axhline(y=WALL_THICKNESS, color='white', linestyle='--', alpha=0.5)
    ax.text(-45, -50, 'Source Region', color='white')
    ax.text(-45, WALL_THICKNESS/2, 'Concrete Wall', color='white')
    ax.text(-45, WALL_THICKNESS + 20, 'Detector Region', color='white')
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

