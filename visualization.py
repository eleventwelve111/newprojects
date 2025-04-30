#!/usr/bin/env python3
"""
Visualization functions for gamma-ray streaming simulation results.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
import seaborn as sns
from pathlib import Path
import pandas as pd
import json

from logging_utils import logger, LogSection, timeit
from config import (RESULTS_DIR, PLOTS_DIR, WALL_THICKNESS, SOURCE_TO_WALL_DISTANCE,
                  CHANNEL_DIAMETERS, SOURCE_ENERGIES, DETECTOR_DISTANCES, DETECTOR_ANGLES)

@timeit
def create_dose_heatmap(results):
    """
    Create a heatmap of dose rates at different distances and angles.
    
    Parameters:
    -----------
    results : list or dict
        Simulation results
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Matplotlib figure
    """
    with LogSection("Creating dose heatmap"):
        # Handle single result vs list
        if isinstance(results, dict):
            results = [results]
            
        # Group results by energy and channel diameter
        grouped_data = {}
        for result in results:
            params = result['parameters']
            energy = params['energy']
            diameter = params['channel_diameter']
            
            if (energy, diameter) not in grouped_data:
                grouped_data[(energy, diameter)] = {
                    'distances': [],
                    'angles': [],
                    'doses': []
                }
            
            grouped_data[(energy, diameter)]['distances'].append(params['detector_distance'])
            grouped_data[(energy, diameter)]['angles'].append(params['detector_angle'])
            grouped_data[(energy, diameter)]['doses'].append(result['total_dose'])
        
        # Create heatmaps for each energy-diameter combination
        figs = {}
        for (energy, diameter), data in grouped_data.items():
            # Convert to arrays
            distances = np.array(data['distances'])
            angles = np.array(data['angles'])
            doses = np.array(data['doses'])
            
            # Get unique values for distances and angles
            unique_distances = np.sort(np.unique(distances))
            unique_angles = np.sort(np.unique(angles))
            
            # Create dose matrix
            dose_matrix = np.zeros((len(unique_distances), len(unique_angles)))
            
            # Fill dose matrix
            for i, dist in enumerate(unique_distances):
                for j, ang in enumerate(unique_angles):
                    mask = (distances == dist) & (angles == ang)
                    if np.any(mask):
                        dose_matrix[i, j] = doses[mask][0]
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Use log scale for dose values
            norm = colors.LogNorm(vmin=max(1e-12, dose_matrix.min()), vmax=dose_matrix.max())
            im = ax.pcolormesh(unique_angles, unique_distances, dose_matrix, norm=norm, cmap='jet')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Dose (rem/hr/source_intensity)', rotation=270, labelpad=20)
            
            # Add labels and title
            ax.set_xlabel('Detector Angle (degrees)')
            ax.set_ylabel('Detector Distance (cm)')
            ax.set_title(f'Dose Heatmap (E={energy} MeV, D={diameter} cm)')
            
            # Save figure
            output_dir = PLOTS_DIR / "dose_heatmaps"
            output_dir.mkdir(exist_ok=True, parents=True)
            output_file = output_dir / f"dose_heatmap_E{energy}_D{diameter}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            
            figs[(energy, diameter)] = fig
            
            logger.info(f"Created dose heatmap for E={energy} MeV, D={diameter} cm")
        
        return figs

@timeit
def create_enhanced_radiation_pattern(results, channel_diameter=None, energy=None):
    """
    Create enhanced visualization of radiation pattern outside the wall.
    
    Parameters:
    -----------
    results : list or dict
        Simulation results
    channel_diameter : float, optional
        Channel diameter to filter results
    energy : float, optional
        Source energy to filter results
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Matplotlib figure
    """
    with LogSection("Creating enhanced radiation pattern visualization"):
        # Handle single result vs list
        if isinstance(results, dict):
            results = [results]
        
        # Find appropriate result
        selected_result = None
        for result in results:
            params = result['parameters']
            if channel_diameter is not None and params['channel_diameter'] != channel_diameter:
                continue
            if energy is not None and params['energy'] != energy:
                continue
            if params['detector_angle'] == 0 and params['detector_distance'] == DETECTOR_DISTANCES[0]:
                selected_result = result
                break
        
        if selected_result is None:
            logger.error("No matching result found for enhanced radiation pattern")
            return None
        
        # Extract fine mesh data if available
        if 'fine_mesh_values' in selected_result:
            mesh_values = np.array(selected_result['fine_mesh_values'])
            mesh_shape = selected_result['fine_mesh_shape']
            mesh_values = mesh_values.reshape(mesh_shape)
        else:
            # Use regular mesh as fallback
            mesh_values = np.array(selected_result['mesh_values'])
            mesh_shape = selected_result['mesh_shape']
            mesh_values = mesh_values.reshape(mesh_shape)
        
        # Take a slice through the center
        central_slice = mesh_values[:, mesh_shape[1]//2, :]
        
        # Create enhanced visualization
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Use log scale for better visualization
        norm = colors.LogNorm(vmin=max(1e-10, central_slice.min()), vmax=central_slice.max())
        
        # Create extent for proper scaling (-20 to 20 cm in x, wall_thickness to wall_thickness+50 in z)
        extent = [-20, 20, WALL_THICKNESS, WALL_THICKNESS + 50]
        im = ax.imshow(central_slice.T, origin='lower', aspect='auto', norm=norm, cmap='inferno',
                      extent=extent)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Flux (particles/cm³/source_particle)', rotation=270, labelpad=20)
        
        # Add labels and title
        params = selected_result['parameters']
        ax.set_xlabel('Horizontal Distance from Channel Axis (cm)')
        ax.set_ylabel('Distance from Back of Wall (cm)')
        ax.set_title(f'Enhanced Radiation Pattern (E={params["energy"]} MeV, D={params["channel_diameter"]} cm)')
        
        # Add wall boundary
        ax.axhline(y=WALL_THICKNESS, color='white', linestyle='-', linewidth=2, alpha=0.7)
        ax.text(-18, WALL_THICKNESS + 2, 'Wall Exit', color='white', fontsize=10)
        
        # Add channel exit marker
        ax.scatter([0], [WALL_THICKNESS], color='cyan', s=100, marker='o', edgecolor='white')
        ax.text(1, WALL_THICKNESS, 'Channel Exit', color='white', fontsize=10)
        
        # Add distance markers
        for distance in [10, 20, 30, 50]:
            y = WALL_THICKNESS + distance
            ax.axhline(y=y, color='white', linestyle='--', alpha=0.3)
            ax.text(-18, y, f'{distance} cm', color='white', fontsize=8)
        
        # Add angle markers
        for angle in [15, 30, 45]:
            # Convert angle to slope (tan)
            rad = np.radians(angle)
            # Draw lines for positive and negative angles
            for sign in [1, -1]:
                x_end = sign * 20
                y_end = WALL_THICKNESS + x_end * np.tan(rad)
                ax.plot([0, x_end], [WALL_THICKNESS, y_end], color='white', linestyle=':', alpha=0.3)
                ax.text(sign * 10, WALL_THICKNESS + sign * 10 * np.tan(rad), f'{angle}°', 
                       color='white', fontsize=8, ha='center')
        
        # Save figure
        output_dir = PLOTS_DIR / "radiation_patterns"
        output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / f"enhanced_pattern_E{params['energy']}_D{params['channel_diameter']}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        logger.info(f"Created enhanced radiation pattern for E={params['energy']} MeV, D={params['channel_diameter']} cm")
        
        return fig

@timeit
def plot_dose_vs_angle(results, energies=None, diameters=None, distances=None):
    """
    Plot dose vs angle at different distances for various energies and channel diameters.
    
    Parameters:
    -----------
    results : list or dict
        Simulation results
    energies : list, optional
        Energies to plot (default is all)
    diameters : list, optional
        Channel diameters to plot (default is all)
    distances : list, optional
        Detector distances to plot (default is all)
    
    Returns:
    --------
    figs : dict
        Dictionary of matplotlib figures
    """
    with LogSection("Plotting dose vs angle"):
        # Handle single result vs list
                # Handle single result vs list
        if isinstance(results, dict):
            results = [results]
        
        # Use defaults if parameters not specified
        if energies is None:
            energies = SOURCE_ENERGIES
        if diameters is None:
            diameters = CHANNEL_DIAMETERS
        if distances is None:
            distances = DETECTOR_DISTANCES
        
        # Convert single values to lists if needed
        if not isinstance(energies, list):
            energies = [energies]
        if not isinstance(diameters, list):
            diameters = [diameters]
        if not isinstance(distances, list):
            distances = [distances]
        
        # Group results by energy, diameter, distance
        grouped_data = {}
        for result in results:
            params = result['parameters']
            energy = params['energy']
            diameter = params['channel_diameter']
            distance = params['detector_distance']
            angle = params['detector_angle']
            
            # Skip if not in requested parameters
            if energy not in energies or diameter not in diameters or distance not in distances:
                continue
            
            key = (energy, diameter, distance)
            if key not in grouped_data:
                grouped_data[key] = {
                    'angles': [],
                    'doses': [],
                    'dose_errors': [] if 'total_dose_error' in result else None
                }
            
            grouped_data[key]['angles'].append(angle)
            grouped_data[key]['doses'].append(result['total_dose'])
            if 'total_dose_error' in result:
                grouped_data[key]['dose_errors'].append(result['total_dose_error'])
        
        # Create plots for each energy and diameter
        figs = {}
        for energy in energies:
            for diameter in diameters:
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot for each distance
                for distance in distances:
                    key = (energy, diameter, distance)
                    if key in grouped_data:
                        data = grouped_data[key]
                        
                        # Sort by angle
                        sort_idx = np.argsort(data['angles'])
                        angles = np.array(data['angles'])[sort_idx]
                        doses = np.array(data['doses'])[sort_idx]
                        
                        label = f"Distance = {distance} cm"
                        ax.semilogy(angles, doses, 'o-', linewidth=2, label=label)
                        
                        # Add error bars if available
                        if data['dose_errors'] is not None:
                            dose_errors = np.array(data['dose_errors'])[sort_idx]
                            ax.fill_between(angles, doses - dose_errors, doses + dose_errors, alpha=0.3)
                
                # Add labels and title
                ax.set_xlabel('Detector Angle (degrees)', fontsize=12)
                ax.set_ylabel('Dose (rem/hr/source_intensity)', fontsize=12)
                ax.set_title(f'Dose vs Angle (E={energy} MeV, Channel D={diameter} cm)', fontsize=14)
                
                # Add grid and legend
                ax.grid(True, which='both', linestyle='--', alpha=0.5)
                ax.legend(fontsize=10)
                
                # Save figure
                key = (energy, diameter)
                figs[key] = fig
                
                output_dir = PLOTS_DIR / "dose_vs_angle"
                output_dir.mkdir(exist_ok=True, parents=True)
                output_file = output_dir / f"dose_vs_angle_E{energy}_D{diameter}.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                
                logger.info(f"Created dose vs angle plot for E={energy} MeV, D={diameter} cm")
        
        return figs

@timeit
def plot_dose_vs_distance(results, energies=None, diameters=None, angles=None):
    """
    Plot dose vs distance for different angles, energies, and channel diameters.
    
    Parameters:
    -----------
    results : list or dict
        Simulation results
    energies : list, optional
        Energies to plot (default is all)
    diameters : list, optional
        Channel diameters to plot (default is all)
    angles : list, optional
        Detector angles to plot (default is all)
    
    Returns:
    --------
    figs : dict
        Dictionary of matplotlib figures
    """
    with LogSection("Plotting dose vs distance"):
        # Handle single result vs list
        if isinstance(results, dict):
            results = [results]
        
        # Use defaults if parameters not specified
        if energies is None:
            energies = SOURCE_ENERGIES
        if diameters is None:
            diameters = CHANNEL_DIAMETERS
        if angles is None:
            angles = DETECTOR_ANGLES
        
        # Convert single values to lists if needed
        if not isinstance(energies, list):
            energies = [energies]
        if not isinstance(diameters, list):
            diameters = [diameters]
        if not isinstance(angles, list):
            angles = [angles]
        
        # Group results by energy, diameter, angle
        grouped_data = {}
        for result in results:
            params = result['parameters']
            energy = params['energy']
            diameter = params['channel_diameter']
            distance = params['detector_distance']
            angle = params['detector_angle']
            
            # Skip if not in requested parameters
            if energy not in energies or diameter not in diameters or angle not in angles:
                continue
            
            key = (energy, diameter, angle)
            if key not in grouped_data:
                grouped_data[key] = {
                    'distances': [],
                    'doses': [],
                    'dose_errors': [] if 'total_dose_error' in result else None
                }
            
            grouped_data[key]['distances'].append(distance)
            grouped_data[key]['doses'].append(result['total_dose'])
            if 'total_dose_error' in result:
                grouped_data[key]['dose_errors'].append(result['total_dose_error'])
        
        # Create plots for each energy and diameter
        figs = {}
        for energy in energies:
            for diameter in diameters:
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot for each angle
                for angle in angles:
                    key = (energy, diameter, angle)
                    if key in grouped_data:
                        data = grouped_data[key]
                        
                        # Sort by distance
                        sort_idx = np.argsort(data['distances'])
                        distances = np.array(data['distances'])[sort_idx]
                        doses = np.array(data['doses'])[sort_idx]
                        
                        label = f"Angle = {angle}°"
                        ax.loglog(distances, doses, 'o-', linewidth=2, label=label)
                        
                        # Add error bars if available
                        if data['dose_errors'] is not None:
                            dose_errors = np.array(data['dose_errors'])[sort_idx]
                            ax.fill_between(distances, doses - dose_errors, doses + dose_errors, alpha=0.3)
                            
                        # Fit a power law (1/r^n) and display
                        if len(distances) > 1:
                            try:
                                # Log-log linear fit for power law
                                log_dist = np.log(distances)
                                log_dose = np.log(doses)
                                coeffs = np.polyfit(log_dist, log_dose, 1)
                                power = coeffs[0]
                                
                                # Add fit line to plot
                                x_fit = np.linspace(min(distances), max(distances), 100)
                                y_fit = np.exp(coeffs[1]) * x_fit**power
                                ax.plot(x_fit, y_fit, '--', linewidth=1, alpha=0.7)
                                
                                # Add power law annotation
                                ax.text(0.05, 0.05 + 0.05*angles.index(angle),
                                        f"{angle}°: ∝ 1/r^{abs(power):.2f}",
                                        transform=ax.transAxes, fontsize=9)
                            except:
                                logger.warning(f"Could not fit power law for angle={angle}°")
                
                # Add reference lines for 1/r and 1/r²
                r_ref = np.linspace(min(DETECTOR_DISTANCES), max(DETECTOR_DISTANCES), 100)
                ref_base = doses[0] * (distances[0] / r_ref)
                ax.plot(r_ref, ref_base, 'k--', alpha=0.3, linewidth=1, label="1/r")
                ax.plot(r_ref, ref_base * (distances[0] / r_ref), 'k:', alpha=0.3, linewidth=1, label="1/r²")
                
                # Add labels and title
                ax.set_xlabel('Detector Distance (cm)', fontsize=12)
                ax.set_ylabel('Dose (rem/hr/source_intensity)', fontsize=12)
                ax.set_title(f'Dose vs Distance (E={energy} MeV, Channel D={diameter} cm)', fontsize=14)
                
                # Add grid and legend
                ax.grid(True, which='both', linestyle='--', alpha=0.5)
                ax.legend(fontsize=10)
                
                # Save figure
                key = (energy, diameter)
                figs[key] = fig
                
                output_dir = PLOTS_DIR / "dose_vs_distance"
                output_dir.mkdir(exist_ok=True, parents=True)
                output_file = output_dir / f"dose_vs_distance_E{energy}_D{diameter}.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                
                logger.info(f"Created dose vs distance plot for E={energy} MeV, D={diameter} cm")
        
        return figs

@timeit
def plot_dose_vs_diameter(results, energies=None, distances=None, angles=None):
    """
    Plot dose vs channel diameter for different energies, distances, and angles.
    
    Parameters:
    -----------
    results : list or dict
        Simulation results
    energies : list, optional
        Energies to plot (default is all)
    distances : list, optional
        Detector distances to plot (default is all)
    angles : list, optional
        Detector angles to plot (default is all)
    
    Returns:
    --------
    figs : dict
        Dictionary of matplotlib figures
    """
    with LogSection("Plotting dose vs channel diameter"):
        # Handle single result vs list
        if isinstance(results, dict):
            results = [results]
        
        # Use defaults if parameters not specified
        if energies is None:
            energies = SOURCE_ENERGIES
        if distances is None:
            distances = DETECTOR_DISTANCES
        if angles is None:
            angles = [0]  # Default to on-axis only
        
        # Convert single values to lists if needed
        if not isinstance(energies, list):
            energies = [energies]
        if not isinstance(distances, list):
            distances = [distances]
        if not isinstance(angles, list):
            angles = [angles]
        
        # Group results by energy, distance, angle
        grouped_data = {}
        for result in results:
            params = result['parameters']
            energy = params['energy']
            diameter = params['channel_diameter']
            distance = params['detector_distance']
            angle = params['detector_angle']
            
            # Skip if not in requested parameters
            if energy not in energies or distance not in distances or angle not in angles:
                continue
            
            key = (energy, distance, angle)
            if key not in grouped_data:
                grouped_data[key] = {
                    'diameters': [],
                    'doses': [],
                    'dose_errors': [] if 'total_dose_error' in result else None
                }
            
            grouped_data[key]['diameters'].append(diameter)
            grouped_data[key]['doses'].append(result['total_dose'])
            if 'total_dose_error' in result:
                grouped_data[key]['dose_errors'].append(result['total_dose_error'])
        
        # Create plots for each energy
        figs = {}
        for energy in energies:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot for each distance (at specified angle)
            for distance in distances:
                for angle in angles:
                    key = (energy, distance, angle)
                    if key in grouped_data:
                        data = grouped_data[key]
                        
                        # Sort by diameter
                        sort_idx = np.argsort(data['diameters'])
                        diameters = np.array(data['diameters'])[sort_idx]
                        doses = np.array(data['doses'])[sort_idx]
                        
                        label = f"Distance = {distance} cm, Angle = {angle}°"
                        ax.loglog(diameters, doses, 'o-', linewidth=2, label=label)
                        
                        # Add error bars if available
                        if data['dose_errors'] is not None:
                            dose_errors = np.array(data['dose_errors'])[sort_idx]
                            ax.fill_between(diameters, doses - dose_errors, doses + dose_errors, alpha=0.3)
                            
                        # Fit a power law (D^n) and display
                        if len(diameters) > 1:
                            try:
                                # Log-log linear fit for power law
                                log_diam = np.log(diameters)
                                log_dose = np.log(doses)
                                coeffs = np.polyfit(log_diam, log_dose, 1)
                                power = coeffs[0]
                                
                                # Add fit line to plot
                                x_fit = np.linspace(min(diameters), max(diameters), 100)
                                y_fit = np.exp(coeffs[1]) * x_fit**power
                                ax.plot(x_fit, y_fit, '--', linewidth=1, alpha=0.7)
                                
                                # Add power law annotation
                                ax.text(0.05, 0.05 + 0.05*distances.index(distance),
                                        f"{distance} cm: ∝ D^{power:.2f}",
                                        transform=ax.transAxes, fontsize=9)
                            except:
                                logger.warning(f"Could not fit power law for distance={distance} cm")
            
            # Add labels and title
            ax.set_xlabel('Channel Diameter (cm)', fontsize=12)
            ax.set_ylabel('Dose (rem/hr/source_intensity)', fontsize=12)
            ax.set_title(f'Dose vs Channel Diameter (E={energy} MeV)', fontsize=14)
            
            # Add grid and legend
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            ax.legend(fontsize=10)
            
            # Save figure
                        # Save figure
            figs[energy] = fig
            
            output_dir = PLOTS_DIR / "dose_vs_diameter"
            output_dir.mkdir(exist_ok=True, parents=True)
            output_file = output_dir / f"dose_vs_diameter_E{energy}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            
            logger.info(f"Created dose vs diameter plot for E={energy} MeV")
        
        return figs

@timeit
def plot_dose_vs_energy(results, diameters=None, distances=None, angles=None):
    """
    Plot dose vs energy for different channel diameters, distances, and angles.
    
    Parameters:
    -----------
    results : list or dict
        Simulation results
    diameters : list, optional
        Channel diameters to plot (default is all)
    distances : list, optional
        Detector distances to plot (default is all)
    angles : list, optional
        Detector angles to plot (default is all)
    
    Returns:
    --------
    figs : dict
        Dictionary of matplotlib figures
    """
    with LogSection("Plotting dose vs energy"):
        # Handle single result vs list
        if isinstance(results, dict):
            results = [results]
        
        # Use defaults if parameters not specified
        if diameters is None:
            diameters = CHANNEL_DIAMETERS
        if distances is None:
            distances = [DETECTOR_DISTANCES[0]]  # Default to first distance
        if angles is None:
            angles = [0]  # Default to on-axis only
        
        # Convert single values to lists if needed
        if not isinstance(diameters, list):
            diameters = [diameters]
        if not isinstance(distances, list):
            distances = [distances]
        if not isinstance(angles, list):
            angles = [angles]
        
        # Group results by diameter, distance, angle
        grouped_data = {}
        for result in results:
            params = result['parameters']
            energy = params['energy']
            diameter = params['channel_diameter']
            distance = params['detector_distance']
            angle = params['detector_angle']
            
            # Skip if not in requested parameters
            if diameter not in diameters or distance not in distances or angle not in angles:
                continue
            
            key = (diameter, distance, angle)
            if key not in grouped_data:
                grouped_data[key] = {
                    'energies': [],
                    'doses': [],
                    'dose_errors': [] if 'total_dose_error' in result else None
                }
            
            grouped_data[key]['energies'].append(energy)
            grouped_data[key]['doses'].append(result['total_dose'])
            if 'total_dose_error' in result:
                grouped_data[key]['dose_errors'].append(result['total_dose_error'])
        
        # Create plots for each selected diameter
        figs = {}
        for diameter in diameters:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot for each distance (at specified angle)
            for distance in distances:
                for angle in angles:
                    key = (diameter, distance, angle)
                    if key in grouped_data:
                        data = grouped_data[key]
                        
                        # Sort by energy
                        sort_idx = np.argsort(data['energies'])
                        energies = np.array(data['energies'])[sort_idx]
                        doses = np.array(data['doses'])[sort_idx]
                        
                        label = f"Distance = {distance} cm, Angle = {angle}°"
                        ax.loglog(energies, doses, 'o-', linewidth=2, label=label)
                        
                        # Add error bars if available
                        if data['dose_errors'] is not None:
                            dose_errors = np.array(data['dose_errors'])[sort_idx]
                            ax.fill_between(energies, doses - dose_errors, doses + dose_errors, alpha=0.3)
            
            # Add labels and title
            ax.set_xlabel('Energy (MeV)', fontsize=12)
            ax.set_ylabel('Dose (rem/hr/source_intensity)', fontsize=12)
            ax.set_title(f'Dose vs Energy (Channel D={diameter} cm)', fontsize=14)
            
            # Add grid and legend
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            ax.legend(fontsize=10)
            
            # Save figure
            figs[diameter] = fig
            
            output_dir = PLOTS_DIR / "dose_vs_energy"
            output_dir.mkdir(exist_ok=True, parents=True)
            output_file = output_dir / f"dose_vs_energy_D{diameter}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            
            logger.info(f"Created dose vs energy plot for D={diameter} cm")
        
        return figs

@timeit
def create_concrete_comparison_plots(comparison_results):
    """
    Create plots comparing different concrete types.
    
    Parameters:
    -----------
    comparison_results : dict
        Dictionary of results for different concrete types
    
    Returns:
    --------
    figs : dict
        Dictionary of matplotlib figures
    """
    with LogSection("Creating concrete comparison plots"):
        concrete_types = list(comparison_results.keys())
        
        # Extract spectrum data for each concrete type
        spectra = {}
        doses = {}
        
        for concrete_type, result in comparison_results.items():
            energy_midpoints = np.array(result['energy_midpoints'])
            flux_spectrum = np.array(result['flux_spectrum'])
            
            # Calculate dose contribution by energy
            dose_vals = np.zeros_like(flux_spectrum)
            for i, energy in enumerate(energy_midpoints):
                dose_vals[i] = flux_spectrum[i] * flux_to_dose_conversion(energy)
            
            spectra[concrete_type] = {
                'energy': energy_midpoints,
                'flux': flux_spectrum
            }
            
            doses[concrete_type] = {
                'energy': energy_midpoints,
                'dose': dose_vals,
                'total_dose': result['total_dose']
            }
        
        # Create spectrum comparison plot
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        for concrete_type in concrete_types:
            energy = spectra[concrete_type]['energy']
            flux = spectra[concrete_type]['flux']
            ax1.loglog(energy, flux, linewidth=2, label=f"{concrete_type.capitalize()} Concrete")
        
        ax1.set_xlabel('Energy (MeV)', fontsize=12)
        ax1.set_ylabel('Flux (particles/cm²/MeV/source_particle)', fontsize=12)
        ax1.set_title('Energy Spectrum Comparison by Concrete Type', fontsize=14)
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        ax1.legend(fontsize=10)
        
        # Create dose contribution plot
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        for concrete_type in concrete_types:
            energy = doses[concrete_type]['energy']
            dose = doses[concrete_type]['dose']
            ax2.semilogx(energy, dose, linewidth=2, label=f"{concrete_type.capitalize()} Concrete")
        
        ax2.set_xlabel('Energy (MeV)', fontsize=12)
        ax2.set_ylabel('Dose Contribution (rem/hr/MeV/source_intensity)', fontsize=12)
        ax2.set_title('Dose Contribution Comparison by Concrete Type', fontsize=14)
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        ax2.legend(fontsize=10)
        
        # Create total dose comparison bar chart
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        
        total_doses = [doses[concrete_type]['total_dose'] for concrete_type in concrete_types]
        concrete_labels = [f"{concrete_type.capitalize()}" for concrete_type in concrete_types]
        
        bars = ax3.bar(concrete_labels, total_doses)
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height,
                    f'{height:.2e}', ha='center', va='bottom', rotation=0, fontsize=10)
        
        ax3.set_xlabel('Concrete Type', fontsize=12)
        ax3.set_ylabel('Total Dose (rem/hr/source_intensity)', fontsize=12)
        ax3.set_title('Total Dose Comparison by Concrete Type', fontsize=14)
        ax3.set_yscale('log')
        ax3.grid(True, which='both', linestyle='--', alpha=0.5, axis='y')
        
        # Save figures
        output_dir = PLOTS_DIR / "concrete_comparison"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        plt.figure(fig1.number)
        plt.savefig(output_dir / "spectrum_comparison.png", dpi=300, bbox_inches='tight')
        
        plt.figure(fig2.number)
        plt.savefig(output_dir / "dose_contribution_comparison.png", dpi=300, bbox_inches='tight')
        
        plt.figure(fig3.number)
        plt.savefig(output_dir / "total_dose_comparison.png", dpi=300, bbox_inches='tight')
        
        logger.info(f"Created concrete comparison plots")
        
        return {
            'spectrum': fig1,
            'dose_contribution': fig2,
            'total_dose': fig3
        }

@timeit
def create_dashboard(results, output_file=None):
    """
    Create a comprehensive dashboard summarizing key results.
    
    Parameters:
    -----------
    results : list
        List of simulation results
    output_file : str or Path, optional
        Output file path
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Matplotlib figure
    """
    with LogSection("Creating results dashboard"):
        # Create large figure for dashboard
        fig = plt.figure(figsize=(24, 18))
        
        # Define grid layout
        gs = plt.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Extract a representative result for detailed view
        reference_result = None
        for result in results:
            params = result['parameters']
            if (params['energy'] == 1.0 and 
                params['channel_diameter'] == 0.5 and
                params['detector_distance'] == 30 and
                params['detector_angle'] == 0):
                reference_result = result
                break
        
        if reference_result is None and results:
            reference_result = results[0]
        
        if reference_result:
            # Extract parameters for reference result
            params = reference_result['parameters']
            
            # Add title
            fig.suptitle(f"Gamma-Ray Streaming Simulation Dashboard\n" +
                        f"Reference: E={params['energy']} MeV, D={params['channel_diameter']} cm, " +
                        f"Distance={params['detector_distance']} cm, Angle={params['detector_angle']}°", 
                        fontsize=20)
            
            # Panel 1: Energy spectrum
            ax1 = fig.add_subplot(gs[0, 0:2])
            energy_midpoints = np.array(reference_result['energy_midpoints'])
            flux_spectrum = np.array(reference_result['flux_spectrum'])
            ax1.loglog(energy_midpoints, flux_spectrum, 'b-', linewidth=2)
            ax1.fill_between(energy_midpoints, 0, flux_spectrum, alpha=0.3, color='blue')
            ax1.set_xlabel('Energy (MeV)')
            ax1.set_ylabel('Flux (particles/cm²/MeV/source)')
            ax1.set_title('Energy Spectrum (Reference Case)')
            ax1.grid(True, which='both', linestyle='--', alpha=0.5)
            
            # Panel 2: Dose vs Distance
            distance_data = {}
            for result in results:
                result_params = result['parameters']
                if (result_params['energy'] == params['energy'] and 
                    result_params['channel_diameter'] == params['channel_diameter'] and
                    result_params['detector_angle'] == 0):
                    distance = result_params['detector_distance']
                    dose = result['total_dose']
                    distance_data[distance] = dose
            
            if distance_data:
                ax2 = fig.add_subplot(gs[0, 2:4])
                distances = sorted(distance_data.keys())
                doses = [distance_data[d] for d in distances]
                ax2.loglog(distances, doses, 'ro-', linewidth=2)
                ax2.set_xlabel('Distance (cm)')
                ax2.set_ylabel('Dose (rem/hr/source)')
                ax2.set_title('Dose vs Distance (On-axis)')
                ax2.grid(True, which='both', linestyle='--', alpha=0.5)
                
                # Add fit line
                if len(distances) > 1:
                    try:
                        # Log-log linear fit for power law
                        log_dist = np.log(distances)
                        log_dose = np.log(doses)
                        coeffs = np.polyfit(log_dist, log_dose, 1)
                        power = coeffs[0]
                        
                        # Add fit line to plot
                        x_fit = np.linspace(min(distances), max(distances), 100)
                        y_fit = np.exp(coeffs[1]) * x_fit**power
                        ax2.plot(x_fit, y_fit, 'r--', linewidth=1, alpha=0.7)
                        
                        # Add power law annotation
                        ax2.text(0.1, 0.9, f"∝ 1/r^{abs(power):.2f}",
                                transform=ax2.transAxes, fontsize=12)
                    except:
                        logger.warning("Could not fit power law for distance")
            
            # Panel 3: Dose vs Angle
            angle_data = {}
            for result in results:
                result_params = result['parameters']
                if (result_params['energy'] == params['energy'] and 
                    result_params['channel_diameter'] == params['channel_diameter'] and
                    result_params['detector_distance'] == params['detector_distance']):
                    angle = result_params['detector_angle']
                    dose = result['total_dose']
                    angle_data[angle] = dose
            
            if angle_data:
                ax3 = fig.add_subplot(gs[1, 0:2])
                angles = sorted(angle_data.keys())
                doses = [angle_data[a] for a in angles]
                ax3.semilogy(angles, doses, 'go-', linewidth=2)
                                ax3.set_xlabel('Angle (degrees)')
                ax3.set_ylabel('Dose (rem/hr/source)')
                ax3.set_title('Dose vs Angle (Fixed Distance)')
                ax3.grid(True, which='both', linestyle='--', alpha=0.5)
            
            # Panel 4: Radiation Pattern
            if 'mesh_values' in reference_result:
                ax4 = fig.add_subplot(gs[1, 2:4])
                mesh_values = np.array(reference_result['mesh_values'])
                mesh_shape = reference_result['mesh_shape']
                mesh_values = mesh_values.reshape(mesh_shape)
                
                # Take central slice
                central_slice = mesh_values[:, mesh_shape[1]//2, :]
                
                # Plot with log scale
                norm = colors.LogNorm(vmin=max(1e-10, central_slice.min()), vmax=central_slice.max())
                im = ax4.imshow(central_slice.T, origin='lower', aspect='auto', norm=norm, cmap='inferno',
                              extent=[-20, 20, WALL_THICKNESS, WALL_THICKNESS + 50])
                
                ax4.set_xlabel('X (cm)')
                ax4.set_ylabel('Z (cm)')
                ax4.set_title('Radiation Pattern (Central Slice)')
                
                # Add colorbar
                plt.colorbar(im, ax=ax4, label='Flux')
            
            # Panel 5: Dose vs Diameter
            diameter_data = {}
            for result in results:
                result_params = result['parameters']
                if (result_params['energy'] == params['energy'] and 
                    result_params['detector_distance'] == params['detector_distance'] and
                    result_params['detector_angle'] == 0):
                    diameter = result_params['channel_diameter']
                    dose = result['total_dose']
                    diameter_data[diameter] = dose
            
            if diameter_data:
                ax5 = fig.add_subplot(gs[2, 0:2])
                diameters = sorted(diameter_data.keys())
                doses = [diameter_data[d] for d in diameters]
                ax5.loglog(diameters, doses, 'mo-', linewidth=2)
                ax5.set_xlabel('Channel Diameter (cm)')
                ax5.set_ylabel('Dose (rem/hr/source)')
                ax5.set_title('Dose vs Channel Diameter')
                ax5.grid(True, which='both', linestyle='--', alpha=0.5)
                
                # Add fit line
                if len(diameters) > 1:
                    try:
                        # Log-log linear fit for power law
                        log_diam = np.log(diameters)
                        log_dose = np.log(doses)
                        coeffs = np.polyfit(log_diam, log_dose, 1)
                        power = coeffs[0]
                        
                        # Add fit line to plot
                        x_fit = np.linspace(min(diameters), max(diameters), 100)
                        y_fit = np.exp(coeffs[1]) * x_fit**power
                        ax5.plot(x_fit, y_fit, 'm--', linewidth=1, alpha=0.7)
                        
                        # Add power law annotation
                        ax5.text(0.1, 0.9, f"∝ D^{power:.2f}",
                                transform=ax5.transAxes, fontsize=12)
                    except:
                        logger.warning("Could not fit power law for diameter")
            
            # Panel 6: Dose vs Energy
            energy_data = {}
            for result in results:
                result_params = result['parameters']
                if (result_params['channel_diameter'] == params['channel_diameter'] and 
                    result_params['detector_distance'] == params['detector_distance'] and
                    result_params['detector_angle'] == 0):
                    energy = result_params['energy']
                    dose = result['total_dose']
                    energy_data[energy] = dose
            
            if energy_data:
                ax6 = fig.add_subplot(gs[2, 2:4])
                energies = sorted(energy_data.keys())
                doses = [energy_data[e] for e in energies]
                ax6.loglog(energies, doses, 'co-', linewidth=2)
                ax6.set_xlabel('Energy (MeV)')
                ax6.set_ylabel('Dose (rem/hr/source)')
                ax6.set_title('Dose vs Energy')
                ax6.grid(True, which='both', linestyle='--', alpha=0.5)
        
        else:
            fig.suptitle("No results available for dashboard", fontsize=20)
        
        # Save figure if output file is specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dashboard saved to {output_path}")
        
        return fig

@timeit
def export_interactive_visualization(results, output_dir=None):
    """
    Export interactive visualization using Plotly or Bokeh.
    
    Parameters:
    -----------
    results : list
        List of simulation results
    output_dir : str or Path, optional
        Output directory
    """
    with LogSection("Exporting interactive visualization"):
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.express as px
            
            # Create output directory if not provided
            if output_dir is None:
                output_dir = PLOTS_DIR / "interactive"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Create dataframe from results
            data = []
            for result in results:
                params = result['parameters']
                row = {
                    'energy': params['energy'],
                    'channel_diameter': params['channel_diameter'],
                    'detector_distance': params['detector_distance'],
                    'detector_angle': params['detector_angle'],
                    'total_dose': result['total_dose'],
                    'total_flux': result.get('total_flux', None)
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # Create interactive 3D plot
            fig = px.scatter_3d(df, x='detector_distance', y='detector_angle', z='total_dose',
                              color='energy', size='channel_diameter',
                              labels={
                                  'detector_distance': 'Distance (cm)',
                                  'detector_angle': 'Angle (degrees)',
                                  'total_dose': 'Dose (rem/hr/source)',
                                  'energy': 'Energy (MeV)',
                                  'channel_diameter': 'Diameter (cm)'
                              },
                              title='Interactive 3D Visualization of Dose vs Distance and Angle')
            
            # Save as HTML
            fig.write_html(output_dir / "interactive_3d.html")
            
            # Create heatmap for each energy-diameter combination
            for energy in df['energy'].unique():
                for diameter in df['channel_diameter'].unique():
                    subset = df[(df['energy'] == energy) & (df['channel_diameter'] == diameter)]
                    
                    if len(subset) > 0:
                        # Prepare data for heatmap
                        pivot = subset.pivot_table(
                            values="total_dose",
                            index="detector_distance",
                            columns="detector_angle"
                        )
                        
                        # Create heatmap
                        fig = px.imshow(pivot, labels=dict(
                            x="Detector Angle (degrees)",
                            y="Detector Distance (cm)",
                            color="Dose (rem/hr/source)"
                        ), title=f"Dose Heatmap (E={energy} MeV, D={diameter} cm)",
                        color_continuous_scale="Viridis")
                        
                        # Save as HTML
                        fig.write_html(output_dir / f"heatmap_E{energy}_D{diameter}.html")
            
            # Save dataframe as CSV for further analysis
            df.to_csv(output_dir / "simulation_results.csv", index=False)
            
            logger.info(f"Interactive visualizations exported to {output_dir}")
            
        except ImportError:
                        logger.warning("Plotly not installed. Skipping interactive visualization.")

@timeit
def create_streaming_analysis_plots(streaming_results):
    """
    Create plots for streaming ratio analysis.
    
    Parameters:
    -----------
    streaming_results : dict
        Dictionary of streaming ratio results
    
    Returns:
    --------
    figs : dict
        Dictionary of matplotlib figures
    """
    with LogSection("Creating streaming analysis plots"):
        figs = {}
        
        # Extract streaming ratios by energy and diameter
        energies = []
        diameters = []
        streaming_ratios = {}
        
        for key, data in streaming_results.items():
            energy, diameter = key
            energies.append(energy)
            diameters.append(diameter)
            
            streaming_ratios[key] = {
                'distances': data['distances'],
                'ratios': data['ratios']
            }
        
        # Get unique values
        energies = sorted(list(set(energies)))
        diameters = sorted(list(set(diameters)))
        
        # Create plots for each energy
        for energy in energies:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot streaming ratio vs distance for each diameter
            for diameter in diameters:
                key = (energy, diameter)
                if key in streaming_ratios:
                    data = streaming_ratios[key]
                    distances = data['distances']
                    ratios = data['ratios']
                    
                    ax.loglog(distances, ratios, 'o-', linewidth=2, label=f"Diameter = {diameter} cm")
            
            # Add labels and title
            ax.set_xlabel('Detector Distance (cm)', fontsize=12)
            ax.set_ylabel('Streaming Ratio (dose with channel / dose without)', fontsize=12)
            ax.set_title(f'Streaming Ratio vs Distance (E={energy} MeV)', fontsize=14)
            
            # Add grid and legend
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            ax.legend(fontsize=10)
            
            # Save figure
            figs[energy] = fig
            
            output_dir = PLOTS_DIR / "streaming_analysis"
            output_dir.mkdir(exist_ok=True, parents=True)
            output_file = output_dir / f"streaming_ratio_E{energy}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        # Create summary plot showing streaming ratio vs diameter for different energies
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # For each energy, plot streaming ratio vs diameter at a reference distance
        reference_distance = min(DETECTOR_DISTANCES)  # Use closest distance
        
        for energy in energies:
            ratio_data = []
            diameter_data = []
            
            for diameter in diameters:
                key = (energy, diameter)
                if key in streaming_ratios:
                    data = streaming_ratios[key]
                    distances = np.array(data['distances'])
                    ratios = np.array(data['ratios'])
                    
                    # Find closest distance to reference
                    idx = np.argmin(np.abs(distances - reference_distance))
                    ratio_data.append(ratios[idx])
                    diameter_data.append(diameter)
            
            if ratio_data:
                ax.loglog(diameter_data, ratio_data, 'o-', linewidth=2, label=f"E={energy} MeV")
        
        # Add labels and title
        ax.set_xlabel('Channel Diameter (cm)', fontsize=12)
        ax.set_ylabel('Streaming Ratio (dose with channel / dose without)', fontsize=12)
        ax.set_title(f'Streaming Ratio vs Diameter (Distance={reference_distance} cm)', fontsize=14)
        
        # Add grid and legend
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend(fontsize=10)
        
        # Save figure
        figs['summary'] = fig
        
        output_dir = PLOTS_DIR / "streaming_analysis"
        output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / "streaming_ratio_summary.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        logger.info("Created streaming analysis plots")
        
        return figs

@timeit
def plot_albedo_analysis(results):
    """
    Create plots for albedo analysis (backscattered radiation).
    
    Parameters:
    -----------
    results : list
        List of simulation results
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Matplotlib figure
    """
    with LogSection("Creating albedo analysis plots"):
        # Extract results with backscatter information
        backscatter_data = {}
        
        for result in results:
            if 'backscatter_flux' in result and 'backscatter_dose' in result:
                params = result['parameters']
                energy = params['energy']
                diameter = params['channel_diameter']
                
                key = (energy, diameter)
                if key not in backscatter_data:
                    backscatter_data[key] = {
                        'flux_ratio': [],
                        'dose_ratio': []
                    }
                
                # Calculate ratios of backscattered to incident radiation
                if result['total_flux'] > 0:
                    flux_ratio = result['backscatter_flux'] / result['total_flux']
                else:
                    flux_ratio = 0
                
                if result['total_dose'] > 0:
                    dose_ratio = result['backscatter_dose'] / result['total_dose']
                else:
                    dose_ratio = 0
                
                backscatter_data[key]['flux_ratio'].append(flux_ratio)
                backscatter_data[key]['dose_ratio'].append(dose_ratio)
        
        if not backscatter_data:
            logger.warning("No backscatter data available for plotting")
            return None
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract unique energies and diameters
        all_keys = list(backscatter_data.keys())
        energies = sorted(list(set([key[0] for key in all_keys])))
        diameters = sorted(list(set([key[1] for key in all_keys])))
        
        # Plot flux ratio
        grouped_data = {energy: [] for energy in energies}
        for energy in energies:
            for diameter in diameters:
                key = (energy, diameter)
                if key in backscatter_data:
                    # Use average if multiple values
                    avg_flux_ratio = np.mean(backscatter_data[key]['flux_ratio'])
                    grouped_data[energy].append((diameter, avg_flux_ratio))
        
        for energy, data_points in grouped_data.items():
            if data_points:
                x, y = zip(*sorted(data_points))
                ax1.semilogx(x, y, 'o-', linewidth=2, label=f"E={energy} MeV")
        
        ax1.set_xlabel('Channel Diameter (cm)', fontsize=12)
        ax1.set_ylabel('Backscatter to Forward Flux Ratio', fontsize=12)
        ax1.set_title('Albedo Analysis - Flux', fontsize=14)
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        ax1.legend(fontsize=10)
        
        # Plot dose ratio
        grouped_data = {energy: [] for energy in energies}
        for energy in energies:
            for diameter in diameters:
                key = (energy, diameter)
                if key in backscatter_data:
                    # Use average if multiple values
                    avg_dose_ratio = np.mean(backscatter_data[key]['dose_ratio'])
                    grouped_data[energy].append((diameter, avg_dose_ratio))
        
        for energy, data_points in grouped_data.items():
            if data_points:
                x, y = zip(*sorted(data_points))
                ax2.semilogx(x, y, 'o-', linewidth=2, label=f"E={energy} MeV")
        
        ax2.set_xlabel('Channel Diameter (cm)', fontsize=12)
        ax2.set_ylabel('Backscatter to Forward Dose Ratio', fontsize=12)
        ax2.set_title('Albedo Analysis - Dose', fontsize=14)
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        output_dir = PLOTS_DIR / "albedo_analysis"
        output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / "albedo_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        logger.info("Created albedo analysis plots")
        
        return fig

@timeit
def create_uncertainty_analysis_plots(results):
    """
    Create plots for uncertainty analysis.
    
    Parameters:
    -----------
    results : list
        List of simulation results
    
    Returns:
    --------
    figs : dict
        Dictionary of matplotlib figures
    """
    with LogSection("Creating uncertainty analysis plots"):
        # Extract results with uncertainty information
        uncertainty_data = {}
        
        for result in results:
            if 'total_dose_error' in result:
                params = result['parameters']
                energy = params['energy']
                diameter = params['channel_diameter']
                distance = params['detector_distance']
                angle = params['detector_angle']
                
                # Calculate relative error
                rel_error = result['total_dose_error'] / result['total_dose'] if result['total_dose'] > 0 else 0
                
                # Store by parameter
                for param_name, param_value in [
                    ('energy', energy),
                    ('diameter', diameter),
                    ('distance', distance),
                    ('angle', angle)
                ]:
                    if param_name not in uncertainty_data:
                        uncertainty_data[param_name] = {}
                    
                    if param_value not in uncertainty_data[param_name]:
                        uncertainty_data[param_name][param_value] = []
                    
                    uncertainty_data[param_name][param_value].append(rel_error)
        
        if not uncertainty_data:
            logger.warning("No uncertainty data available for plotting")
            return {}
        
        # Create figures
        figs = {}
        
        # Plot relative error vs each parameter
        for param_name, param_data in uncertainty_data.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Calculate average relative error for each parameter value
            param_values = []
            rel_errors = []
            std_errors = []
            
            for param_value, errors in param_data.items():
                param_values.append(param_value)
                rel_errors.append(np.mean(errors))
                std_errors.append(np.std(errors))
            
            # Sort by parameter value
            sorted_idx = np.argsort(param_values)
            param_values = np.array(param_values)[sorted_idx]
            rel_errors = np.array(rel_errors)[sorted_idx]
            std_errors = np.array(std_errors)[sorted_idx]
            
            # Plot
            ax.errorbar(param_values, rel_errors, yerr=std_errors, fmt='o-', linewidth=2)
            
            # Convert parameter name to label
            param_labels = {
                'energy': 'Energy (MeV)',
                'diameter': 'Channel Diameter (cm)',
                'distance': 'Detector Distance (cm)',
                'angle': 'Detector Angle (degrees)'
            }
            
            ax.set_xlabel(param_labels.get(param_name, param_name), fontsize=12)
            ax.set_ylabel('Relative Error', fontsize=12)
            ax.set_title(f'Relative Error vs {param_labels.get(param_name, param_name)}', fontsize=14)
            
            # Use log scale for x-axis if appropriate
            if param_name in ['energy', 'diameter', 'distance']:
                ax.set_xscale('log')
            
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            
            # Save figure
            figs[param_name] = fig
            
            output_dir = PLOTS_DIR / "uncertainty_analysis"
            output_dir.mkdir(exist_ok=True, parents=True)
            output_file = output_dir / f"uncertainty_vs_{param_name}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        logger.info("Created uncertainty analysis plots")
        
        return figs
def generate_spectrum_plots(results):
    """
    Generate detailed plots of energy spectra.
    
    Parameters:
    -----------
    results : list or dict
        Simulation results
    
    Returns:
    --------
    figs : dict
        Dictionary of matplotlib figures
    """
    with LogSection("Generating spectrum plots"):
        # Handle single result vs list
        if isinstance(results, dict):
            results = [results]
        
        # Group results by energy and channel diameter
        grouped_data = {}
        for result in results:
            if 'energy_midpoints' not in result or 'flux_spectrum' not in result:
                continue
                
            params = result['parameters']
            energy = params['energy']
            diameter = params['channel_diameter']
            distance = params['detector_distance']
            angle = params['detector_angle']
            
            key = (energy, diameter)
            if key not in grouped_data:
                grouped_data[key] = []
            
            grouped_data[key].append({
                'distance': distance,
                'angle': angle,
                'energy_midpoints': np.array(result['energy_midpoints']),
                'flux_spectrum': np.array(result['flux_spectrum'])
            })
        
        # Create spectrum plots for each energy-diameter combination
        figs = {}
        for (energy, diameter), data_list in grouped_data.items():
            # Plot spectra for different distances (for on-axis points)
            on_axis_data = [d for d in data_list if d['angle'] == 0]
            
            if on_axis_data:
                # Sort by distance
                on_axis_data.sort(key=lambda x: x['distance'])
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for data in on_axis_data:
                    dist = data['distance']
                    energy_points = data['energy_midpoints']
                    flux = data['flux_spectrum']
                    
                    ax.loglog(energy_points, flux, linewidth=2, label=f"Distance = {dist} cm")
                
                # Add labels and title
                ax.set_xlabel('Energy (MeV)', fontsize=12)
                ax.set_ylabel('Flux (particles/cm²/MeV/source)', fontsize=12)
                ax.set_title(f'Energy Spectrum (E={energy} MeV, D={diameter} cm)', fontsize=14)
                ax.grid(True, which='both', linestyle='--', alpha=0.5)
                ax.legend(fontsize=10)
                
                # Save figure
                figs[(energy, diameter, 'distance')] = fig
                
                output_dir = PLOTS_DIR / "spectrum_analysis"
                output_dir.mkdir(exist_ok=True, parents=True)
                output_file = output_dir / f"spectrum_by_distance_E{energy}_D{diameter}.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
            
            # Plot spectra for different angles (at closest distance)
            closest_distance = min([d['distance'] for d in data_list])
            angle_data = [d for d in data_list if d['distance'] == closest_distance]
            
            if angle_data and len(angle_data) > 1:
                # Sort by angle
                angle_data.sort(key=lambda x: x['angle'])
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for data in angle_data:
                    ang = data['angle']
                    energy_points = data['energy_midpoints']
                    flux = data['flux_spectrum']
                    
                    ax.loglog(energy_points, flux, linewidth=2, label=f"Angle = {ang}°")
                
                # Add labels and title
                ax.set_xlabel('Energy (MeV)', fontsize=12)
                ax.set_ylabel('Flux (particles/cm²/MeV/source)', fontsize=12)
                ax.set_title(f'Energy Spectrum (E={energy} MeV, D={diameter} cm, Dist={closest_distance} cm)', 
                           fontsize=14)
                ax.grid(True, which='both', linestyle='--', alpha=0.5)
                ax.legend(fontsize=10)
                
                # Save figure
                figs[(energy, diameter, 'angle')] = fig
                
                output_dir = PLOTS_DIR / "spectrum_analysis"
                output_dir.mkdir(exist_ok=True, parents=True)
                output_file = output_dir / f"spectrum_by_angle_E{energy}_D{diameter}.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        logger.info(f"Generated {len(figs)} spectrum plots")
        return figs

@timeit
def generate_all_visualizations(results, output_base_dir=None):
    """
    Generate all visualization plots for the simulation results.
    
    Parameters:
    -----------
    results : list or dict
        Simulation results
    output_base_dir : str or Path, optional
        Base output directory
    
    Returns:
    --------
    plot_data : dict
        Dictionary containing all generated plots
    """
    with LogSection("Generating all visualizations"):
        if output_base_dir is None:
            output_base_dir = PLOTS_DIR
        else:
            output_base_dir = Path(output_base_dir)
        
        # Make sure output directory exists
        output_base_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate all plots
        plot_data = {}
        
        # Dose heatmaps
        logger.info("Generating dose heatmaps...")
        plot_data['dose_heatmaps'] = create_dose_heatmap(results)
        
        # Enhanced radiation pattern
        logger.info("Generating enhanced radiation patterns...")
        plot_data['radiation_patterns'] = create_enhanced_radiation_pattern(results)
        
        # Dose vs angle
        logger.info("Generating dose vs angle plots...")
        plot_data['dose_vs_angle'] = plot_dose_vs_angle(results)
        
        # Dose vs distance
        logger.info("Generating dose vs distance plots...")
        plot_data['dose_vs_distance'] = plot_dose_vs_distance(results)
        
        # Dose vs diameter
        logger.info("Generating dose vs diameter plots...")
        plot_data['dose_vs_diameter'] = plot_dose_vs_diameter(results)
        
        # Dose vs energy
        logger.info("Generating dose vs energy plots...")
        plot_data['dose_vs_energy'] = plot_dose_vs_energy(results)
        
        # Spectrum plots
        logger.info("Generating spectrum plots...")
        plot_data['spectrum_plots'] = generate_spectrum_plots(results)
        
        # Uncertainty analysis
        logger.info("Generating uncertainty analysis plots...")
        plot_data['uncertainty_plots'] = create_uncertainty_analysis_plots(results)
        
        # Dashboard
        logger.info("Generating dashboard...")
        dashboard_file = output_base_dir / "dashboard.png"
        plot_data['dashboard'] = create_dashboard(results, dashboard_file)
        
        # Interactive visualization
        logger.info("Generating interactive visualizations...")
        export_interactive_visualization(results, output_base_dir / "interactive")
        
        # Generate index.html for easy navigation
        generate_visualization_index(output_base_dir)
        
        logger.info(f"All visualizations generated in {output_base_dir}")
        
        return plot_data

def generate_visualization_index(output_dir):
    """
    Generate an HTML index file for navigating visualization results.
    
    Parameters:
    -----------
    output_dir : Path
        Directory containing visualization outputs
    """
    with LogSection("Generating visualization index"):
        # Find all PNG and HTML files in the directory tree
        plot_files = []
        for ext in ['.png', '.html']:
            plot_files.extend(list(output_dir.glob(f"**/*{ext}")))
        
        # Sort files by directory then name
        plot_files.sort(key=lambda x: (x.parent.name, x.name))
        
        # Group files by directory
        grouped_files = {}
        for file in plot_files:
            rel_path = file.relative_to(output_dir)
            dir_name = rel_path.parent
            if dir_name not in grouped_files:
                grouped_files[dir_name] = []
            grouped_files[dir_name].append(rel_path)
        
        # Generate HTML
        html = ["<!DOCTYPE html>",
                "<html>",
                "<head>",
                "    <title>Gamma-Ray Streaming Visualization Results</title>",
                "    <style>",
                "        body { font-family: Arial, sans-serif; margin: 20px; }",
                "        h1 { color: #2c3e50; }",
                "        h2 { color: #3498db; margin-top: 30px; }",
                "        .thumbnail { max-width: 200px; max-height: 200px; margin: 10px; }",
                "        .plot-container { display: flex; flex-wrap: wrap; }",
                "        .plot-item { margin: 10px; text-align: center; }",
                "        .nav { position: fixed; top: 0; width: 200px; height: 100%; overflow: auto; padding: 10px; }",
                "        .content { margin-left: 220px; }",
                "    </style>",
                "</head>",
                "<body>",
                "    <div class='nav'>",
                "        <h3>Navigation</h3>",
                "        <ul>"]
        
        # Add navigation links
        for dir_name in grouped_files.keys():
            if dir_name == Path('.'):
                dir_text = "Main Directory"
            else:
                dir_text = str(dir_name)
            html.append(f"            <li><a href='#{dir_name}'>{dir_text}</a></li>")
        
        html.append("        </ul>",
                    "    </div>",
                    "    <div class='content'>",
                    "        <h1>Gamma-Ray Streaming Visualization Results</h1>")
        
        # Add content by directory
        for dir_name, files in grouped_files.items():
            if dir_name == Path('.'):
                dir_text = "Main Directory"
            else:
                dir_text = str(dir_name)
            
            html.append(f"        <h2 id='{dir_name}'>{dir_text}</h2>")
            html.append("        <div class='plot-container'>")
            
            for file in files:
                file_path = str(file).replace('\\', '/')
                file_name = file.name
                
                # Different handling for HTML files
                if file.suffix.lower() == '.html':
                    html.append(f"            <div class='plot-item'>")
                    html.append(f"                <a href='{file_path}' target='_blank'>")
                    html.append(f"                    <div>{file_name}</div>")
                    html.append(f"                    <div>Interactive Plot</div>")
                    html.append(f"                </a>")
                    html.append(f"            </div>")
                else:
                    html.append(f"            <div class='plot-item'>")
                    html.append(f"                <a href='{file_path}' target='_blank'>")
                    html.append(f"                    <img src='{file_path}' class='thumbnail'>")
                    html.append(f"                    <div>{file_name}</div>")
                    html.append(f"                </a>")
                    html.append(f"            </div>")
            
            html.append("        </div>")
        
        html.append("    </div>",
                   "</body>",
                   "</html>")
        
        # Write the HTML file
        index_path = output_dir / "index.html"
        with open(index_path, 'w') as f:
            f.write('\n'.join(html))
        
        logger.info(f"Visualization index generated at {index_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate visualizations for gamma-ray streaming results")
    parser.add_argument("--results", type=str, required=True, help="Path to results JSON file")
    parser.add_argument("--output", type=str, default=None, help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Load results
    with open(args.results, 'r') as f:
        results = json.load(f)
    
    # Generate all visualizations
    generate_all_visualizations(results, args.output)




