#!/usr/bin/env python3
"""
Spectrum analysis for gamma-ray streaming simulation.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, interpolate, optimize
import pandas as pd
from pathlib import Path

from logging_utils import logger, LogSection, timeit
from config import (RESULTS_DIR, PLOTS_DIR, SOURCE_ENERGIES, 
                  DETECTOR_DISTANCES, DETECTOR_ANGLES)
from dose import flux_to_dose_conversion

@timeit
def calculate_spectrum_metrics(spectrum_data):
    """
    Calculate various metrics from energy spectrum data.
    
    Parameters:
    -----------
    spectrum_data : dict
        Dictionary containing 'energy_midpoints' and 'flux_spectrum'
    
    Returns:
    --------
    metrics : dict
        Dictionary of spectrum metrics
    """
    with LogSection("Calculating spectrum metrics"):
        # Extract data
        energy_midpoints = np.array(spectrum_data['energy_midpoints'])
        flux_spectrum = np.array(spectrum_data['flux_spectrum'])
        
        # Calculate total flux
        total_flux = np.sum(flux_spectrum)
        
        # Calculate mean energy
        if total_flux > 0:
            mean_energy = np.sum(energy_midpoints * flux_spectrum) / total_flux
        else:
            mean_energy = 0.0
        
        # Calculate dose contribution
        dose_contribution = np.zeros_like(flux_spectrum)
        for i, energy in enumerate(energy_midpoints):
            dose_contribution[i] = flux_spectrum[i] * flux_to_dose_conversion(energy)
        
        total_dose = np.sum(dose_contribution)
        
        # Calculate dose-weighted mean energy
        if total_dose > 0:
            dose_weighted_mean_energy = np.sum(energy_midpoints * dose_contribution) / total_dose
        else:
            dose_weighted_mean_energy = 0.0
        
        # Find the peak energy (energy with maximum flux)
        peak_index = np.argmax(flux_spectrum)
        peak_energy = energy_midpoints[peak_index]
        
        # Calculate the full width at half maximum (FWHM)
        half_max = flux_spectrum[peak_index] / 2.0
        
        # Find energies where flux is above half maximum
        above_half_max = flux_spectrum >= half_max
        
        # Create interpolation function for more accurate FWHM
        if np.any(above_half_max):
            # Get indices where spectrum crosses half maximum
            crossing_indices = np.where(np.diff(above_half_max.astype(int)))[0]
            
            if len(crossing_indices) >= 2:
                # For each crossing, interpolate to find exact energy where flux = half_max
                crossing_energies = []
                for idx in crossing_indices:
                    # Make sure we don't go beyond array boundaries
                    if idx + 1 < len(flux_spectrum):
                        e1, e2 = energy_midpoints[idx], energy_midpoints[idx + 1]
                        f1, f2 = flux_spectrum[idx], flux_spectrum[idx + 1]
                        
                        # Linear interpolation
                        if f1 != f2:  # Avoid division by zero
                            e_crossing = e1 + (half_max - f1) * (e2 - e1) / (f2 - f1)
                            crossing_energies.append(e_crossing)
                
                if len(crossing_energies) >= 2:
                    fwhm = max(crossing_energies) - min(crossing_energies)
                else:
                    # Fallback if interpolation fails
                    energy_above = energy_midpoints[above_half_max]
                    fwhm = energy_above[-1] - energy_above[0] if len(energy_above) > 0 else 0.0
            else:
                # If there's only one crossing or none
                energy_above = energy_midpoints[above_half_max]
                fwhm = energy_above[-1] - energy_above[0] if len(energy_above) > 0 else 0.0
        else:
            fwhm = 0.0
        
        # Hardening ratio (ratio of flux above 1 MeV to total flux)
        high_energy_mask = energy_midpoints > 1.0
        if total_flux > 0:
            hardening_ratio = np.sum(flux_spectrum[high_energy_mask]) / total_flux
        else:
            hardening_ratio = 0.0
        
        # Calculate metrics
        metrics = {
                        'total_flux': float(total_flux),
            'mean_energy': float(mean_energy),
            'peak_energy': float(peak_energy),
            'fwhm': float(fwhm),
            'total_dose': float(total_dose),
            'dose_weighted_mean_energy': float(dose_weighted_mean_energy),
            'hardening_ratio': float(hardening_ratio)
        }
        
        logger.info(f"Spectrum metrics calculated:")
        logger.info(f"  Total flux: {total_flux:.4e} particles/cm²/source_particle")
        logger.info(f"  Mean energy: {mean_energy:.4f} MeV")
        logger.info(f"  Peak energy: {peak_energy:.4f} MeV")
        logger.info(f"  FWHM: {fwhm:.4f} MeV")
        logger.info(f"  Total dose: {total_dose:.4e} rem/hr/source_intensity")
        
        return metrics

@timeit
def plot_energy_spectrum(spectrum_data, title=None, output_file=None):
    """
    Plot energy spectrum from simulation results.
    
    Parameters:
    -----------
    spectrum_data : dict
        Dictionary containing 'energy_midpoints' and 'flux_spectrum'
    title : str, optional
        Plot title
    output_file : str or Path, optional
        Output file path
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Matplotlib figure
    """
    with LogSection("Plotting energy spectrum"):
        # Extract data
        energy_midpoints = np.array(spectrum_data['energy_midpoints'])
        flux_spectrum = np.array(spectrum_data['flux_spectrum'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot spectrum
        ax.semilogx(energy_midpoints, flux_spectrum, 'b-', linewidth=2)
        ax.fill_between(energy_midpoints, 0, flux_spectrum, alpha=0.3, color='blue')
        
        # Add labels and title
        ax.set_xlabel('Energy (MeV)', fontsize=12)
        ax.set_ylabel('Flux (particles/cm²/MeV/source_particle)', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('Energy Spectrum', fontsize=14)
        
        # Add grid
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Save figure if output file is specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Spectrum plot saved to {output_path}")
        
        return fig

@timeit
def analyze_spectrum_trends(results_list):
    """
    Analyze trends in spectra across different parameters.
    
    Parameters:
    -----------
    results_list : list
        List of simulation result dictionaries
    
    Returns:
    --------
    trends : dict
        Dictionary of spectrum trend analyses
    """
    with LogSection("Analyzing spectrum trends"):
        # Initialize trends dictionary
        trends = {
            'energy_dependence': {},
            'diameter_dependence': {},
            'distance_dependence': {},
            'angle_dependence': {}
        }
        
        # Group results by parameters
        by_energy = {}
        by_diameter = {}
        by_distance = {}
        by_angle = {}
        
        for result in results_list:
            params = result['parameters']
            energy = params['energy']
            diameter = params['channel_diameter']
            distance = params['detector_distance']
            angle = params['detector_angle']
            
            # Calculate metrics for this result
            metrics = calculate_spectrum_metrics(result)
            
            # Store metrics by parameter
            if energy not in by_energy:
                by_energy[energy] = []
            by_energy[energy].append(metrics)
            
            if diameter not in by_diameter:
                by_diameter[diameter] = []
            by_diameter[diameter].append(metrics)
            
            if distance not in by_distance:
                by_distance[distance] = []
            by_distance[distance].append(metrics)
            
            if angle not in by_angle:
                by_angle[angle] = []
            by_angle[angle].append(metrics)
        
        # Analyze energy dependence (at fixed diameter, distance, angle)
        reference_diameter = CHANNEL_DIAMETERS[0]
        reference_distance = DETECTOR_DISTANCES[0]
        reference_angle = DETECTOR_ANGLES[0]
        
        energy_trend_data = []
        for energy in SOURCE_ENERGIES:
            # Find results with matching reference parameters
            matching_results = []
            for result in results_list:
                params = result['parameters']
                if (params['energy'] == energy and 
                    params['channel_diameter'] == reference_diameter and
                    params['detector_distance'] == reference_distance and
                    params['detector_angle'] == reference_angle):
                    matching_results.append(result)
            
            if matching_results:
                # Use the first matching result
                metrics = calculate_spectrum_metrics(matching_results[0])
                energy_trend_data.append({
                    'energy': energy,
                    'metrics': metrics
                })
        
        trends['energy_dependence'] = {
            'reference_parameters': {
                'diameter': reference_diameter,
                'distance': reference_distance,
                'angle': reference_angle
            },
            'trend_data': energy_trend_data
        }
        
        # Similar analyses for other parameters...
        # (diameter dependence, distance dependence, angle dependence)
        
        # Analyze spread vs energy
        # For each energy, look at how flux falls off with angle
        energy_spread_data = []
        for energy in SOURCE_ENERGIES:
            angle_data = []
            for angle in DETECTOR_ANGLES:
                matching_results = []
                for result in results_list:
                    params = result['parameters']
                    if (params['energy'] == energy and 
                        params['channel_diameter'] == reference_diameter and
                        params['detector_distance'] == reference_distance and
                        params['detector_angle'] == angle):
                        matching_results.append(result)
                
                if matching_results:
                    metrics = calculate_spectrum_metrics(matching_results[0])
                    angle_data.append({
                        'angle': angle,
                        'total_flux': metrics['total_flux'],
                        'total_dose': metrics['total_dose']
                    })
            
            if angle_data:
                # Calculate how quickly flux falls off with angle
                angles = np.array([item['angle'] for item in angle_data])
                fluxes = np.array([item['total_flux'] for item in angle_data])
                
                # Normalize to on-axis flux
                if fluxes[0] > 0:
                    normalized_fluxes = fluxes / fluxes[0]
                    
                    # Find angle where flux drops to 50% and 10%
                    if len(angles) > 1:
                        try:
                            flux_50_angle = np.interp(0.5, normalized_fluxes[::-1], angles[::-1])
                            flux_10_angle = np.interp(0.1, normalized_fluxes[::-1], angles[::-1])
                        except:
                            flux_50_angle = np.nan
                            flux_10_angle = np.nan
                    else:
                        flux_50_angle = np.nan
                        flux_10_angle = np.nan
                else:
                    flux_50_angle = np.nan
                    flux_10_angle = np.nan
                
                energy_spread_data.append({
                    'energy': energy,
                    'angle_data': angle_data,
                    'flux_50_angle': float(flux_50_angle),
                    'flux_10_angle': float(flux_10_angle)
                })
        
        trends['energy_spread_analysis'] = energy_spread_data
        
        logger.info(f"Spectrum trend analysis complete")
        
        return trends

@timeit
def plot_spectrum_comparison(results_list, parameter='energy', fixed_params=None, output_file=None):
    """
    Plot spectrum comparison across different values of a parameter.
    
    Parameters:
    -----------
    results_list : list
        List of simulation result dictionaries
    parameter : str
        Parameter to vary ('energy', 'diameter', 'distance', 'angle')
    fixed_params : dict, optional
        Fixed parameter values for other parameters
    output_file : str or Path, optional
        Output file path
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Matplotlib figure
    """
    with LogSection(f"Plotting spectrum comparison by {parameter}"):
        # Default fixed parameters if not provided
        if fixed_params is None:
            fixed_params = {
                'energy': SOURCE_ENERGIES[0],
                'channel_diameter': CHANNEL_DIAMETERS[0],
                'detector_distance': DETECTOR_DISTANCES[0],
                'detector_angle': DETECTOR_ANGLES[0]
            }
            
            # Remove the parameter we're varying
            if parameter in fixed_params:
                del fixed_params[parameter]
        
        # Find parameter values to compare
        if parameter == 'energy':
            values = SOURCE_ENERGIES
            param_name = 'energy'
            param_label = 'Energy (MeV)'
        elif parameter == 'diameter':
            values = CHANNEL_DIAMETERS
            param_name = 'channel_diameter'
            param_label = 'Channel Diameter (cm)'
        elif parameter == 'distance':
            values = DETECTOR_DISTANCES
            param_name = 'detector_distance'
            param_label = 'Detector Distance (cm)'
        elif parameter == 'angle':
            values = DETECTOR_ANGLES
            param_name = 'detector_angle'
            param_label = 'Detector Angle (degrees)'
        else:
            raise ValueError(f"Unknown parameter: {parameter}")
        
        # Group results by parameter value
        grouped_results = {}
        for value in values:
            matching_results = []
            for result in results_list:
                params = result['parameters']
                matches = params[param_name] == value
                
                # Check if fixed parameters match
                for fixed_param, fixed_value in fixed_params.items():
                    matches = matches and (params[fixed_param] == fixed_value)
                
                if matches:
                    matching_results.append(result)
            
            if matching_results:
                grouped_results[value] = matching_results[0]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot spectra for each parameter value
        for value, result in grouped_results.items():
            energy_midpoints = np.array(result['energy_midpoints'])
            flux_spectrum = np.array(result['flux_spectrum'])
            
            # Plot spectrum
            label = f"{param_label} = {value}"
            ax.loglog(energy_midpoints, flux_spectrum, linewidth=2, label=label)
        
        # Add labels and title
        ax.set_xlabel('Energy (MeV)', fontsize=12)
        ax.set_ylabel('Flux (particles/cm²/MeV/source_particle)', fontsize=12)
        
        fixed_param_str = ', '.join([f"{k}={v}" for k, v in fixed_params.items()])
        ax.set_title(f'Energy Spectrum Comparison by {param_label}\nFixed Parameters: {fixed_param_str}', fontsize=14)
        
        # Add grid and legend
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend(fontsize=10)
        
        # Save figure if output file is specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Spectrum comparison plot saved to {output_path}")
        
        return fig

