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
            
        # Create output directory
        output_dir = PLOTS_DIR / "heatmaps"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Group by energy and channel diameter
        energies = sorted(list(set([r['energy'] for r in results])))
        diameters = sorted(list(set([r['channel_diameter'] for r in results])))
        
        # Create a separate heatmap for each energy and diameter combination
        for energy in energies:
            for diameter in diameters:
                # Filter results for this energy and diameter
                filtered_results = [r for r in results 
                                  if abs(r['energy'] - energy) < 0.01 
                                  and abs(r['channel_diameter'] - diameter) < 0.01]
                
                if not filtered_results:
                    continue
                
                # Extract distances and angles
                distances = sorted(list(set([r['detector_distance'] for r in filtered_results])))
                angles = sorted(list(set([r['detector_angle'] for r in filtered_results])))
                
                # Create 2D array of dose values
                dose_matrix = np.zeros((len(distances), len(angles)))
                
                for i, dist in enumerate(distances):
                    for j, angle in enumerate(angles):
                        # Find matching result
                        matches = [r for r in filtered_results 
                                 if abs(r['detector_distance'] - dist) < 0.01 
                                 and abs(r['detector_angle'] - angle) < 0.01]
                        
                        if matches:
                            dose_matrix[i, j] = matches[0]['dose']['value']
                
                # Create heatmap
                plt.figure(figsize=(10, 8))
                
                # Use logarithmic color scale
                norm = colors.LogNorm(vmin=max(dose_matrix.min(), 1e-10), vmax=max(dose_matrix.max(), 1e-9))
                
                heatmap = sns.heatmap(dose_matrix, cmap='viridis', norm=norm,
                                     xticklabels=angles, yticklabels=distances,
                                     annot=True, fmt='.2e', annot_kws={'size': 8})
                
                plt.title(f'Dose Rate (rem/hr) - E={energy} MeV, Ø={diameter} cm')
                plt.xlabel('Detector Angle (degrees)')
                plt.ylabel('Detector Distance (cm)')
                
                # Add colorbar
                cbar = heatmap.collections[0].colorbar
                cbar.set_label('Dose Rate (rem/hr)')
                
                # Save figure
                filename = f"heatmap_E{energy}_D{diameter}.png"
                plt.tight_layout()
                plt.savefig(output_dir / filename, dpi=300)
                plt.close()
                logger.info(f"Created heatmap: {filename}")
        
        # Create summary heatmap for each energy (at 0 degrees)
        for energy in energies:
            # Filter results for this energy and 0 degrees
            filtered_results = [r for r in results 
                              if abs(r['energy'] - energy) < 0.01 
                              and abs(r['detector_angle']) < 0.01]
            
            if not filtered_results:
                continue
            
            # Extract distances and diameters
            distances = sorted(list(set([r['detector_distance'] for r in filtered_results])))
            plot_diameters = sorted(list(set([r['channel_diameter'] for r in filtered_results])))
            
            # Create 2D array of dose values
            dose_matrix = np.zeros((len(distances), len(plot_diameters)))
            
            for i, dist in enumerate(distances):
                for j, diam in enumerate(plot_diameters):
                    # Find matching result
                    matches = [r for r in filtered_results 
                             if abs(r['detector_distance'] - dist) < 0.01 
                             and abs(r['channel_diameter'] - diam) < 0.01]
                    
                    if matches:
                        dose_matrix[i, j] = matches[0]['dose']['value']
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            
            # Use logarithmic color scale
            norm = colors.LogNorm(vmin=max(dose_matrix.min(), 1e-10), vmax=max(dose_matrix.max(), 1e-9))
            
            heatmap = sns.heatmap(dose_matrix, cmap='viridis', norm=norm,
                                 xticklabels=[f"{d:.2f} cm" for d in plot_diameters], 
                                 yticklabels=distances,
                                 annot=True, fmt='.2e', annot_kws={'size': 8})
            
            plt.title(f'Dose Rate vs Channel Diameter (E={energy} MeV, Angle=0°)')
            plt.xlabel('Channel Diameter')
            plt.ylabel('Detector Distance (cm)')
            
            # Add colorbar
            cbar = heatmap.collections[0].colorbar
            cbar.set_label('Dose Rate (rem/hr)')
            
            # Save figure
            filename = f"heatmap_summary_E{energy}.png"
            plt.tight_layout()
            plt.savefig(output_dir / filename, dpi=300)
            plt.close()
            logger.info(f"Created summary heatmap: {filename}")
        
        return plt.figure()  # Return a placeholder figure

@timeit
def plot_dose_vs_angle(results):
    """
    Plot dose vs angle for different energies, channel diameters, and distances.
    
    Parameters:
    -----------
    results : list or dict
        Simulation results
    
    Returns:
    --------
    figs : list
        List of matplotlib figures
    """
    with LogSection("Creating dose vs angle plots"):
        # Handle single result vs list
        if isinstance(results, dict):
            results = [results]
            
        # Create output directory
        output_dir = PLOTS_DIR / "angular"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Group by energy and distance
        energies = sorted(list(set([r['energy'] for r in results])))
        distances = sorted(list(set([r['detector_distance'] for r in results])))
        
        figs = []
        
        # Create a separate plot for each energy and distance combination
        for energy in energies:
            for distance in distances:
                # Filter results for this energy and distance
                filtered_results = [r for r in results 
                                  if abs(r['energy'] - energy) < 0.01 
                                  and abs(r['detector_distance'] - distance) < 0.01]
                
                if not filtered_results:
                    continue
                
                # Extract diameters and angles
                diameters = sorted(list(set([r['channel_diameter'] for r in filtered_results])))
                angles = sorted(list(set([r['detector_angle'] for r in filtered_results])))
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot dose vs angle for each diameter
                for diameter in diameters:
                    # Extract dose values for this diameter
                    diam_results = [r for r in filtered_results 
                                  if abs(r['channel_diameter'] - diameter) < 0.01]
                    
                    # Sort by angle
                    diam_results.sort(key=lambda r: r['detector_angle'])
                    
                    plot_angles = [r['detector_angle'] for r in diam_results]
                    plot_doses = [r['dose']['value'] for r in diam_results]
                    
                    ax.plot(plot_angles, plot_doses, 'o-', label=f'Ø = {diameter:.3f} cm')
                
                ax.set_title(f'Dose vs Angle (E={energy} MeV, Distance={distance} cm)')
                ax.set_xlabel('Detector Angle (degrees)')
                ax.set_ylabel('Dose Rate (rem/hr)')
                ax.set_yscale('log')
                ax.grid(True, which='both', linestyle='--', alpha=0.7)
                ax.legend()
                
                # Save figure
                filename = f"dose_vs_angle_E{energy}_dist{distance}.png"
                plt.tight_layout()
                plt.savefig(output_dir / filename, dpi=300)
                logger.info(f"Created angular plot: {filename}")
                
                figs.append(fig)
        
        # Create combined plots for each energy
        for energy in energies:
            # Filter results for this energy
            energy_results = [r for r in results if abs(r['energy'] - energy) < 0.01]
            
            if not energy_results:
                continue
            
            # Extract unique diameters
            diameters = sorted(list(set([r['channel_diameter'] for r in energy_results])))
            
            # Create a figure with subplots for different diameters
            fig, axes = plt.subplots(len(diameters), 1, figsize=(10, 5*len(diameters)), sharex=True)
            
            if len(diameters) == 1:
                axes = [axes]  # Make axes iterable if only one subplot
            
            for i, diameter in enumerate(diameters):
                # Filter results for this diameter
                diam_results = [r for r in energy_results 
                              if abs(r['channel_diameter'] - diameter) < 0.01]
                
                # Group by distance
                for distance in distances:
                    # Filter by distance
                    dist_results = [r for r in diam_results 
                                  if abs(r['detector_distance'] - distance) < 0.01]
                    
                    if not dist_results:
                        continue
                    
                    # Sort by angle
                    dist_results.sort(key=lambda r: r['detector_angle'])
                    
                    plot_angles = [r['detector_angle'] for r in dist_results]
                    plot_doses = [r['dose']['value'] for r in dist_results]
                    
                    axes[i].plot(plot_angles, plot_doses, 'o-', label=f'Dist = {distance} cm')
                
                axes[i].set_title(f'Channel Diameter = {diameter:.3f} cm')
                axes[i].set_ylabel('Dose Rate (rem/hr)')
                axes[i].set_yscale('log')
                axes[i].grid(True, which='both', linestyle='--', alpha=0.7)
                axes[i].legend()
            
            # Set common labels
            fig.suptitle(f'Dose vs Angle for Different Distances (E={energy} MeV)', fontsize=16)
            axes[-1].set_xlabel('Detector Angle (degrees)')
            
            # Save figure
            filename = f"combined_dose_vs_angle_E{energy}.png"
            plt.tight_layout()
            plt.savefig(output_dir / filename, dpi=300)
            plt.close()
            logger.info(f"Created combined angular plot: {filename}")
            
            figs.append(fig)
        
        return figs

@timeit
def plot_energy_spectrum(results):
    """
    Plot energy spectrum for different configurations.
    
    Parameters:
    -----------
    results : list or dict
        Simulation results
    
    Returns:
    --------
    figs : list
        List of matplotlib figures
    """
    with LogSection("Creating energy spectrum plots"):
        # Handle single result vs list
        if isinstance(results, dict):
            results = [results]
            
        # Create output directory
        output_dir = PLOTS_DIR / "spectrum"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        figs = []
        
        # Process only results that contain energy spectrum data
        spectrum_results = [r for r in results if 'energy_spectrum' in r]
        
        if not spectrum_results:
            logger.warning("No energy spectrum data found in results")
            return figs
        
        # Group by energy and channel diameter
        energies = sorted(list(set([r['energy'] for r in spectrum_results])))
        diameters = sorted(list(set([r['channel_diameter'] for r in spectrum_results])))
        
        # Plot spectrum for each source energy and channel diameter
        for energy in energies:
            for diameter in diameters:
                # Filter results
                filtered_results = [r for r in spectrum_results 
                                  if abs(r['energy'] - energy) < 0.01 
                                  and abs(r['channel_diameter'] - diameter) < 0.01]
                
                if not filtered_results:
                    continue
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot spectrum for different detector positions
                for result in filtered_results:
                    dist = result['detector_distance']
                    angle = result['detector_angle']
                    
                    # Extract spectrum data
                    energy_midpoints = result['energy_spectrum']['energy_midpoints']
                    flux_spectrum = result['energy_spectrum']['flux_spectrum']
                    
                                        ax.plot(energy_midpoints, flux_spectrum, 
                           label=f'Dist={dist} cm, Angle={angle}°')
                
                ax.set_title(f'Energy Spectrum (Source: {energy} MeV, Channel Ø: {diameter} cm)')
                ax.set_xlabel('Energy (MeV)')
                ax.set_ylabel('Flux per unit energy (1/cm²-s-MeV)')
                ax.set_yscale('log')
                ax.set_xscale('log')
                ax.grid(True, which='both', linestyle='--', alpha=0.7)
                ax.legend()
                
                # Save figure
                filename = f"spectrum_E{energy}_D{diameter}.png"
                plt.tight_layout()
                plt.savefig(output_dir / filename, dpi=300)
                plt.close()
                logger.info(f"Created energy spectrum plot: {filename}")
                
                figs.append(fig)
        
        # Create a combined plot showing energy-dependent angular distribution
        if spectrum_results:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Extract a subset of results for clarity - use 1.0 MeV source, 30 cm distance, various angles
            subset = [r for r in spectrum_results 
                    if abs(r['energy'] - 1.0) < 0.01
                    and abs(r['detector_distance'] - 30) < 0.01
                    and r['channel_diameter'] > 0.4]  # Use larger diameter for better statistics
            
            if subset:
                # Sort by angle
                subset.sort(key=lambda r: r['detector_angle'])
                
                # Create a colormap for angles
                angles = [r['detector_angle'] for r in subset]
                cmap = plt.cm.viridis
                norm = plt.Normalize(min(angles), max(angles))
                
                for i, result in enumerate(subset):
                    angle = result['detector_angle']
                    color = cmap(norm(angle))
                    
                    # Extract spectrum data
                    energy_midpoints = result['energy_spectrum']['energy_midpoints']
                    flux_spectrum = result['energy_spectrum']['flux_spectrum']
                    
                    ax.plot(energy_midpoints, flux_spectrum, color=color,
                           label=f'Angle={angle}°')
                
                ax.set_title('Energy Spectrum at Different Angles (1.0 MeV Source, 30 cm Distance)')
                ax.set_xlabel('Energy (MeV)')
                ax.set_ylabel('Flux per unit energy (1/cm²-s-MeV)')
                ax.set_yscale('log')
                ax.set_xscale('log')
                ax.grid(True, which='both', linestyle='--', alpha=0.7)
                
                # Create a colorbar to show angle mapping
                sm = ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label('Detector Angle (degrees)')
                
                # Alternative legend for specific angles
                if len(angles) <= 6:
                    ax.legend()
                
                # Save figure
                filename = "energy_angular_distribution.png"
                plt.tight_layout()
                plt.savefig(output_dir / filename, dpi=300)
                plt.close()
                logger.info(f"Created energy angular distribution plot: {filename}")
                
                figs.append(fig)
        
        return figs

@timeit
def plot_comparison_dose_calc_vs_simulation(results):
    """
    Compare analytical dose calculations with simulation results.
    
    Parameters:
    -----------
    results : list or dict
        Simulation results
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Matplotlib figure
    """
    with LogSection("Creating dose calculation comparison plot"):
        # Handle single result vs list
        if isinstance(results, dict):
            results = [results]
            
        # Create output directory
        output_dir = PLOTS_DIR / "validation"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Check if results contain analytical calculations
        has_analytical = any(['analytical_dose' in r for r in results])
        
        if not has_analytical:
            logger.warning("No analytical dose calculations found in results")
            return None
        
        # Filter results with analytical calculations
        comparison_results = [r for r in results if 'analytical_dose' in r]
        
        # Group by energy
        energies = sorted(list(set([r['energy'] for r in comparison_results])))
        
        # Create comparison plots for each energy
        figs = []
        
        for energy in energies:
            # Filter by energy
            energy_results = [r for r in comparison_results if abs(r['energy'] - energy) < 0.01]
            
            if not energy_results:
                continue
            
            # Plot simulated vs analytical doses
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Extract data for plotting
            simulated_doses = [r['dose']['value'] for r in energy_results]
            analytical_doses = [r['analytical_dose'] for r in energy_results]
            
            # Create scatter plot
            ax.scatter(analytical_doses, simulated_doses, alpha=0.7)
            
            # Add identity line
            min_dose = min(min(simulated_doses), min(analytical_doses))
            max_dose = max(max(simulated_doses), max(analytical_doses))
            dose_range = [min_dose*0.9, max_dose*1.1]
            ax.plot(dose_range, dose_range, 'k--', label='Perfect Match')
            
            # Add best fit line
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                np.log10(analytical_doses), np.log10(simulated_doses))
            
            best_fit_label = f'Best Fit (Power: {slope:.2f}, R²: {r_value**2:.3f})'
            ax.plot(dose_range, [10**(intercept + slope*np.log10(x)) for x in dose_range], 
                   'r-', label=best_fit_label)
            
            # Label points with diameter and distance
            for i, r in enumerate(energy_results):
                ax.annotate(f"Ø={r['channel_diameter']:.2f}, d={r['detector_distance']}",
                          (analytical_doses[i], simulated_doses[i]),
                          fontsize=8, alpha=0.7)
            
            ax.set_title(f'Simulated vs. Analytical Dose (E={energy} MeV)')
            ax.set_xlabel('Analytical Dose (rem/hr)')
            ax.set_ylabel('Simulated Dose (rem/hr)')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, which='both', linestyle='--', alpha=0.7)
            ax.legend()
            
            # Save figure
            filename = f"sim_vs_calc_E{energy}.png"
            plt.tight_layout()
            plt.savefig(output_dir / filename, dpi=300)
            plt.close()
            logger.info(f"Created comparison plot: {filename}")
            
            figs.append(fig)
        
        # Create a combined comparison across energies
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract all data
        simulated_doses = [r['dose']['value'] for r in comparison_results]
        analytical_doses = [r['analytical_dose'] for r in comparison_results]
        energies_list = [r['energy'] for r in comparison_results]
        
        # Create scatter plot with color coding by energy
        cmap = plt.cm.viridis
        norm = plt.Normalize(min(energies_list), max(energies_list))
        sc = ax.scatter(analytical_doses, simulated_doses, 
                      c=energies_list, cmap=cmap, norm=norm, alpha=0.7)
        
        # Add identity line
        min_dose = min(min(simulated_doses), min(analytical_doses))
        max_dose = max(max(simulated_doses), max(analytical_doses))
        dose_range = [min_dose*0.9, max_dose*1.1]
        ax.plot(dose_range, dose_range, 'k--', label='Perfect Match')
        
        # Add best fit line
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            np.log10(analytical_doses), np.log10(simulated_doses))
        
        best_fit_label = f'Best Fit (Power: {slope:.2f}, R²: {r_value**2:.3f})'
        ax.plot(dose_range, [10**(intercept + slope*np.log10(x)) for x in dose_range], 
               'r-', label=best_fit_label)
        
        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Energy (MeV)')
        
        ax.set_title('Simulated vs. Analytical Dose (All Energies)')
        ax.set_xlabel('Analytical Dose (rem/hr)')
        ax.set_ylabel('Simulated Dose (rem/hr)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.legend()
        
        # Save figure
        filename = "sim_vs_calc_all_energies.png"
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300)
        plt.close()
        logger.info(f"Created combined comparison plot: {filename}")
        
        figs.append(fig)
        
        return figs

@timeit
def plot_radiation_pattern(results):
    """
    Plot the radiation pattern in 2D space.
    
    Parameters:
    -----------
    results : list or dict
        Simulation results
    
    Returns:
    --------
    figs : list
        List of matplotlib figures
    """
    with LogSection("Creating radiation pattern plots"):
        # Handle single result vs list
        if isinstance(results, dict):
            results = [results]
            
        # Create output directory
        output_dir = PLOTS_DIR / "streaming"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Group by energy and channel diameter
        energies = sorted(list(set([r['energy'] for r in results])))
        diameters = sorted(list(set([r['channel_diameter'] for r in results])))
        
        figs = []
        
        # Create polar plots for each energy and diameter combination
        for energy in energies:
            for diameter in diameters:
                # Filter results
                filtered_results = [r for r in results 
                                  if abs(r['energy'] - energy) < 0.01 
                                  and abs(r['channel_diameter'] - diameter) < 0.01]
                
                if not filtered_results:
                    continue
                
                # Create polar plot
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='polar')
                
                # Group by distance
                distances = sorted(list(set([r['detector_distance'] for r in filtered_results])))
                
                # Create a colormap for distances
                cmap = plt.cm.viridis
                norm = plt.Normalize(min(distances), max(distances))
                
                # Plot each distance as a separate line
                for distance in distances:
                    # Get results for this distance
                    dist_results = [r for r in filtered_results 
                                  if abs(r['detector_distance'] - distance) < 0.01]
                    
                    # Sort by angle
                    dist_results.sort(key=lambda r: r['detector_angle'])
                    
                    # Convert degrees to radians
                    angles_rad = np.radians([r['detector_angle'] for r in dist_results])
                    
                    # Add the 0 angle to close the circle if needed
                    if len(angles_rad) > 1 and min(angles_rad) > 0:
                        angles_rad = np.append(angles_rad, 0)
                        dist_results.append(dist_results[0])  # Duplicate first point
                    
                    # Extract dose values
                    dose_values = [r['dose']['value'] for r in dist_results]
                    
                    # Normalize for better visualization
                    max_dose = max([r['dose']['value'] for r in filtered_results])
                    normalized_values = [d / max_dose for d in dose_values]
                    
                    # Plot line
                    color = cmap(norm(distance))
                    ax.plot(angles_rad, normalized_values, 'o-', color=color, label=f'{distance} cm')
                
                ax.set_title(f'Radiation Pattern (E={energy} MeV, Ø={diameter} cm)')
                ax.set_theta_zero_location('N')  # 0 degrees at the top
                ax.set_theta_direction(-1)  # clockwise
                ax.set_rlabel_position(45)  # Move radial labels away from plotted line
                ax.set_xticks(np.radians([0, 15, 30, 45, 60, 75, 90]))
                ax.set_xticklabels(['0°', '15°', '30°', '45°', '60°', '75°', '90°'])
                
                # Add colorbar
                sm = ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label('Detector Distance (cm)')
                
                # Save figure
                filename = f"radiation_pattern_E{energy}_D{diameter}.png"
                plt.tight_layout()
                plt.savefig(output_dir / filename, dpi=300)
                plt.close()
                logger.info(f"Created radiation pattern plot: {filename}")
                
                figs.append(fig)
        
        # Create a summary plot with multiple energies for the same diameter
        for diameter in diameters:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='polar')
            
            # Filter results for 30 cm distance and this diameter
            filtered_results = [r for r in results 
                              if abs(r['detector_distance'] - 30) < 0.01
                              and abs(r['channel_diameter'] - diameter) < 0.01]
            
            if not filtered_results:
                continue
            
            # Create a colormap for energies
            plot_energies = sorted(list(set([r['energy'] for r in filtered_results])))
            cmap = plt.cm.plasma
            norm = plt.Normalize(min(plot_energies), max(plot_energies))
            # Plot each energy as a separate line
            for energy in plot_energies:
                # Get results for this energy
                energy_results = [r for r in filtered_results 
                               if abs(r['energy'] - energy) < 0.01]
                
                # Sort by angle
                energy_results.sort(key=lambda r: r['detector_angle'])
                
                # Convert degrees to radians
                angles_rad = np.radians([r['detector_angle'] for r in energy_results])
                
                # Add the 0 angle to close the circle if needed
                if len(angles_rad) > 1 and min(angles_rad) > 0:
                    angles_rad = np.append(angles_rad, 0)
                    energy_results.append(energy_results[0])  # Duplicate first point
                
                # Extract dose values
                dose_values = [r['dose']['value'] for r in energy_results]
                
                # Normalize for better visualization
                max_dose = max([r['dose']['value'] for r in energy_results])
                normalized_values = [d / max_dose for d in dose_values]
                
                # Plot line
                color = cmap(norm(energy))
                ax.plot(angles_rad, normalized_values, 'o-', color=color, label=f'{energy} MeV')
            
            ax.set_title(f'Radiation Pattern for Different Energies (Ø={diameter} cm, Distance=30 cm)')
            ax.set_theta_zero_location('N')  # 0 degrees at the top
            ax.set_theta_direction(-1)  # clockwise
            ax.set_rlabel_position(45)  # Move radial labels away from plotted line
            ax.set_xticks(np.radians([0, 15, 30, 45, 60, 75, 90]))
            ax.set_xticklabels(['0°', '15°', '30°', '45°', '60°', '75°', '90°'])
            
            # Add colorbar
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Photon Energy (MeV)')
            
            # Save figure
            filename = f"radiation_pattern_summary_D{diameter}.png"
            plt.tight_layout()
            plt.savefig(output_dir / filename, dpi=300)
            plt.close()
            logger.info(f"Created radiation pattern summary plot: {filename}")
            
            figs.append(fig)
        
        # Create a final composite visualization showing key patterns
        # This will be a 2x2 grid showing radiation patterns for different scenarios
        fig, axes = plt.subplots(2, 2, figsize=(16, 14), subplot_kw={'projection': 'polar'})
        
        # Plot scenarios
        scenarios = [
            # (energy, diameter, title)
            (0.5, min(diameters), "Low Energy, Small Channel"),
            (0.5, max(diameters), "Low Energy, Large Channel"),
            (5.0, min(diameters), "High Energy, Small Channel"),
            (5.0, max(diameters), "High Energy, Large Channel")
        ]
        
        for i, (energy, diameter, title) in enumerate(scenarios):
            ax = axes[i//2, i%2]
            
            # Filter results
            scenario_results = [r for r in results 
                             if abs(r['energy'] - energy) < 0.1 
                             and abs(r['channel_diameter'] - diameter) < 0.01]
            
            # Group by distance
            plot_distances = sorted(list(set([r['detector_distance'] for r in scenario_results])))
            cmap = plt.cm.viridis
            norm = plt.Normalize(min(plot_distances) if plot_distances else 0, 
                                max(plot_distances) if plot_distances else 100)
            
            if not scenario_results:
                ax.set_title(f"{title}\n(No data available)")
                continue
            
            # Plot each distance
            for distance in plot_distances:
                # Get results for this distance
                dist_results = [r for r in scenario_results 
                              if abs(r['detector_distance'] - distance) < 0.01]
                
                # Sort by angle
                dist_results.sort(key=lambda r: r['detector_angle'])
                
                # Convert degrees to radians
                angles_rad = np.radians([r['detector_angle'] for r in dist_results])
                
                # Add the 0 angle to close the circle if needed
                if len(angles_rad) > 1 and min(angles_rad) > 0:
                    angles_rad = np.append(angles_rad, 0)
                    dist_results.append(dist_results[0])  # Duplicate first point
                
                # Extract dose values
                dose_values = [r['dose']['value'] for r in dist_results]
                
                # Normalize for better visualization
                max_dose = max([r['dose']['value'] for r in dist_results])
                normalized_values = [d / max_dose for d in dose_values]
                
                # Plot line
                color = cmap(norm(distance))
                ax.plot(angles_rad, normalized_values, 'o-', color=color)
            
            ax.set_title(title)
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_xticks(np.radians([0, 30, 60, 90]))
            ax.set_xticklabels(['0°', '30°', '60°', '90°'])
        
        # Add a common colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist())
        cbar.set_label('Detector Distance (cm)')
        
        fig.suptitle('Radiation Streaming Patterns: Energy and Channel Size Effects', fontsize=16)
        
        # Save figure
        filename = "radiation_pattern_summary.png"
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300)
        plt.close()
        logger.info(f"Created radiation pattern composite plot: {filename}")
        
        figs.append(fig)
        
        return figs

@timeit
def plot_dose_vs_channel_diameter(results):
    """
    Plot dose vs channel diameter for different energies and distances.
    
    Parameters:
    -----------
    results : list or dict
        Simulation results
    
    Returns:
    --------
    figs : list
        List of matplotlib figures
    """
    with LogSection("Creating dose vs channel diameter plots"):
        # Handle single result vs list
        if isinstance(results, dict):
            results = [results]
            
        # Create output directory
        output_dir = PLOTS_DIR / "diameter"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Group by energy and distance (for on-axis points)
        filtered_results = [r for r in results if abs(r['detector_angle']) < 0.1]
        
        energies = sorted(list(set([r['energy'] for r in filtered_results])))
        distances = sorted(list(set([r['detector_distance'] for r in filtered_results])))
        
        figs = []
        
        # Create plots for each energy
        for energy in energies:
            # Filter by energy
            energy_results = [r for r in filtered_results if abs(r['energy'] - energy) < 0.01]
            
            if not energy_results:
                continue
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot dose vs diameter for each distance
            for distance in distances:
                # Filter by distance
                dist_results = [r for r in energy_results if abs(r['detector_distance'] - distance) < 0.01]
                
                if not dist_results:
                    continue
                
                # Sort by diameter
                dist_results.sort(key=lambda r: r['channel_diameter'])
                
                diameters = [r['channel_diameter'] for r in dist_results]
                doses = [r['dose']['value'] for r in dist_results]
                
                ax.plot(diameters, doses, 'o-', label=f'Distance = {distance} cm')
            
            ax.set_title(f'Dose vs Channel Diameter (E={energy} MeV, Angle=0°)')
            ax.set_xlabel('Channel Diameter (cm)')
            ax.set_ylabel('Dose Rate (rem/hr)')
            ax.set_yscale('log')
            ax.grid(True, which='both', linestyle='--', alpha=0.7)
            ax.legend()
            
            # Save figure
            filename = f"dose_vs_diameter_E{energy}.png"
            plt.tight_layout()
            plt.savefig(output_dir / filename, dpi=300)
            plt.close()
            logger.info(f"Created diameter plot: {filename}")
            
            figs.append(fig)
        
        # Create a combined plot with all energies for 30 cm distance
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter for 30 cm distance
        dist_30cm_results = [r for r in filtered_results if abs(r['detector_distance'] - 30) < 0.1]
        
        if dist_30cm_results:
            # Group by energy
            for energy in energies:
                # Filter by energy
                energy_results = [r for r in dist_30cm_results if abs(r['energy'] - energy) < 0.01]
                
                if not energy_results:
                    continue
                
                # Sort by diameter
                energy_results.sort(key=lambda r: r['channel_diameter'])
                
                diameters = [r['channel_diameter'] for r in energy_results]
                doses = [r['dose']['value'] for r in energy_results]
                
                ax.plot(diameters, doses, 'o-', label=f'E = {energy} MeV')
            
            ax.set_title('Dose vs Channel Diameter (Distance=30 cm, Angle=0°)')
            ax.set_xlabel('Channel Diameter (cm)')
            ax.set_ylabel('Dose Rate (rem/hr)')
            ax.set_yscale('log')
            ax.grid(True, which='both', linestyle='--', alpha=0.7)
            ax.legend()
            
            # Save figure
            filename = "dose_vs_diameter_all_energies.png"
            plt.tight_layout()
            plt.savefig(output_dir / filename, dpi=300)
            plt.close()
            logger.info(f"Created combined diameter plot: {filename}")
            
            figs.append(fig)
        
        return figs

@timeit
def create_all_visualizations(results):
    """
    Create all visualizations for the results.
    
    Parameters:
    -----------
    results : list or dict
        Simulation results
    
    Returns:
    --------
    viz_results : dict
        Dictionary of visualization results
    """
    with LogSection("Creating all visualizations"):
        viz_results = {}
        
        # Create heatmaps
        viz_results['heatmaps'] = create_dose_heatmap(results)
        
        # Create angular plots
        viz_results['angular_plots'] = plot_dose_vs_angle(results)
        
        # Create energy spectrum plots
        viz_results['spectrum_plots'] = plot_energy_spectrum(results)
        
        # Create dose vs channel diameter plots
        viz_results['diameter_plots'] = plot_dose_vs_channel_diameter(results)
        
        # Create radiation pattern plots
        viz_results['pattern_plots'] = plot_radiation_pattern(results)
        
        # Create comparison plots
        viz_results['comparison_plots'] = plot_comparison_dose_calc_vs_simulation(results)
        
        return viz_results

