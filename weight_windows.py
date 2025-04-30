#!/usr/bin/env python3
"""
Weight window generation and application for variance reduction in OpenMC.
"""
import numpy as np
import openmc
import matplotlib.pyplot as plt
from logging_utils import logger, LogSection, timeit

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
        
        # Create a mesh for weight windows
        ww_mesh = openmc.RegularMesh()
        ww_mesh.dimension = [20, 20, 30]
        
        # Set mesh boundaries to cover the problem geometry
        ww_mesh.lower_left = [-50, -50, -150]
        ww_mesh.upper_right = [50, 50, 150]
        
        # Create a mesh filter
        ww_mesh_filter = openmc.MeshFilter(ww_mesh)
        
        # Create a tally to score particle importances
        ww_tally = openmc.Tally(name="weight_window_tally")
        ww_tally.filters = [ww_mesh_filter]
        ww_tally.scores = ['flux']
        
        # Create tallies collection
        tallies = openmc.Tallies([ww_tally])
        
        # Create settings for weight window generation
        settings = openmc.Settings()
        settings.source = source
        settings.run_mode = 'fixed source'
        settings.particles = int(particles)
        settings.batches = 10
        
        # Create the model
        model = openmc.Model(geometry=geometry, materials=materials, settings=settings, tallies=tallies)
        
        # Run a short simulation to get flux distribution
        logger.info(f"Running initial simulation with {particles} particles for weight window generation")
        sp_filename = model.run()
        
        # Open the statepoint file
        with openmc.StatePoint(sp_filename) as sp:
            # Get the tally for weight window generation
            tally = sp.get_tally(name="weight_window_tally")
            
            # Extract flux values from tally
            flux_values = tally.get_values(scores=['flux']).flatten()
            
            # Replace zeros/NaNs with small positive values to avoid division by zero
            flux_values = np.nan_to_num(flux_values, nan=1e-10, posinf=1e10, neginf=1e-10)
            flux_values = np.maximum(flux_values, 1e-10)
            
            # Normalize flux values (importance) relative to source region
            # Identify source region (assume it's near the beginning of the mesh)
            source_region_indices = np.where(flux_values > 0.5 * np.max(flux_values))[0][:5]
            source_importance = np.mean(flux_values[source_region_indices])
            
            # Calculate relative importance
            importance = flux_values / source_importance
            
            # Calculate weight window bounds
            # Lower weight bound is proportional to 1/importance
            lower_bounds = 1.0 / importance
            
            # Apply bounds to avoid extreme values
            lower_bounds = np.clip(lower_bounds, 1e-5, 1e5)
            
            # Upper bounds are typically 5-10 times the lower bounds
            upper_bounds = lower_bounds * 5.0
            
            # Survival weight is typically the average of lower and upper bounds
            survival_weights = 0.5 * (lower_bounds + upper_bounds)
            
            # Store weight window parameters
            weight_windows = {
                'mesh': ww_mesh,
                'lower_bounds': lower_bounds,
                'upper_bounds': upper_bounds,
                'survival_weights': survival_weights,
                'importance': importance
            }
            
            logger.info(f"Generated weight windows with {len(flux_values)} mesh cells")
            logger.info(f"Min importance: {np.min(importance):.3e}, Max importance: {np.max(importance):.3e}")
            
            return weight_windows

@timeit
def apply_weight_windows(model, weight_windows):
    """
    Apply weight windows to a model for variance reduction.
    
    Parameters:
    -----------
    model : dict
        Dictionary containing model components
    weight_windows : dict
        Weight window parameters
    
    Returns:
    --------
    updated_model : dict
        Updated model with variance reduction
    """
    with LogSection("Applying weight windows"):
        # Extract model components
        materials = model['materials']
        geometry = model['geometry']
        tallies = model.get('tallies', None)
        settings = model.get('settings', None)
        
        if settings is None:
            settings = openmc.Settings()
        
        # Create weight window mesh
        ww_mesh = weight_windows['mesh']
        lower_bounds = weight_windows['lower_bounds']
        upper_bounds = weight_windows['upper_bounds']
        survival_weights = weight_windows['survival_weights']
        
        # Create weight window object
        weight_window = openmc.WeightWindows(
            mesh=ww_mesh,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            survival_weights=survival_weights
        )
        
        # Add weight window to settings
        settings.weight_windows = [weight_window]
        
        # Set weight cutoff to be less than the minimum lower weight bound
        min_weight = np.min(lower_bounds) * 0.1
        settings.weight_cutoff = (min_weight, 0.5)
        
        # Enable survival biasing
        settings.survival_biasing = True
        
        # Update the model
        updated_model = model.copy()
        updated_model['settings'] = settings
        
        logger.info("Applied weight windows to model")
        logger.info(f"Set weight cutoff to ({min_weight:.3e}, 0.5)")
        
        return updated_model

def plot_weight_windows(weight_windows, slice_dim='z', slice_index=15):
    """
    Plot weight windows for a specific slice.
    
    Parameters:
    -----------
    weight_windows : dict
        Weight window parameters
    slice_dim : str
        Dimension to slice ('x', 'y', or 'z')
    slice_index : int
        Index of the slice to plot
    
    Returns:
    --------
    fig : matplotlib.Figure
        Figure containing the weight window plot
    """
    # Extract mesh dimensions
    mesh = weight_windows['mesh']
    nx, ny, nz = mesh.dimension
    
    # Extract importance values
    importance = weight_windows['importance']
    
    # Reshape importance to mesh dimensions
    importance_3d = importance.reshape(nx, ny, nz)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract the slice based on dimension
    if slice_dim == 'x':
        if slice_index >= nx:
            slice_index = nx // 2
        slice_data = importance_3d[slice_index, :, :]
        extent = [mesh.lower_left[2], mesh.upper_right[2], 
                  mesh.lower_left[1], mesh.upper_right[1]]
        xlabel = 'Z'
        ylabel = 'Y'
    elif slice_dim == 'y':
        if slice_index >= ny:
            slice_index = ny // 2
        slice_data = importance_3d[:, slice_index, :]
        extent = [mesh.lower_left[2], mesh.upper_right[2], 
                  mesh.lower_left[0], mesh.upper_right[0]]
        xlabel = 'Z'
        ylabel = 'X'
    else:  # z
        if slice_index >= nz:
            slice_index = nz // 2
        slice_data = importance_3d[:, :, slice_index]
        extent = [mesh.lower_left[0], mesh.upper_right[0], 
                  mesh.lower_left[1], mesh.upper_right[1]]
        xlabel = 'X'
        ylabel = 'Y'
    
    # Plot importance as a color map
    im = ax.imshow(slice_data.T, origin='lower', extent=extent, 
                   cmap='viridis', norm=plt.LogNorm())
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Relative Importance')
    
    # Add labels and title
    ax.set_xlabel(f'{xlabel} (cm)')
    ax.set_ylabel(f'{ylabel} (cm)')
    ax.set_title(f'Weight Window Importance - {slice_dim.upper()} Slice {slice_index}')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    return fig
