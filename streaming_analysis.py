#!/usr/bin/env python3
"""
Streaming analysis for gamma-ray penetration through concrete shields with air channels.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, interpolate, optimize
import pandas as pd
from pathlib import Path

from logging_utils import logger, LogSection, timeit
from config import (RESULTS_DIR, PLOTS_DIR, SOURCE_ENERGIES, WALL_THICKNESS,
                  SOURCE_TO_WALL_DISTANCE, DETECTOR_DISTANCES, DETECTOR_ANGLES)

@timeit
def calculate_streaming_ratio(results):
    """
    Calculate streaming ratio as the dose with channel divided by dose without channel.
    
    Parameters:
    -----------
    results : list
        List of simulation result dictionaries
    
    Returns:
    --------
    streaming_ratios : dict
        Dictionary of streaming ratio results
    """
    with LogSection("Calculating streaming ratios"):
        # Group results by energy and distance (only on-axis points)
        on_axis_results = [r for r in results if abs(r['detector_angle']) < 0.1]
        
        streaming_ratios = {}
        
        for energy in set(r['energy'] for r in on_axis_results):
            energy_results = [r for r in on_axis_results if abs(r['energy'] - energy) < 0.01]
            
            for distance in set(r['detector_distance'] for r in energy_results):
                dist_results = [r for r in energy_results if abs(r['detector_distance'] - distance) < 0.1]
                
                # Find no-channel case (assume diameter=0 or very small)
                no_channel_cases = [r for r in dist_results if r['channel_diameter'] < 0.01]
                if not no_channel_cases:
                    # Use the smallest diameter as reference
                    no_channel_cases = [min(dist_results, key=lambda r: r['channel_diameter'])]
                
                if not no_channel_cases:
                    logger.warning(f"No reference (no-channel) case found for E={energy} MeV, d={distance} cm")
                    continue
                
                ref_dose = no_channel_cases[0]['dose']['value']
                
                # Calculate ratio for each diameter
                for result in dist_results:
                    if result == no_channel_cases[0]:
                        ratio = 1.0  # By definition
                    else:
                        ratio = result['dose']['value'] / ref_dose
                    
                    key = f"E{energy}_D{result['channel_diameter']}_d{distance}"
                    streaming_ratios[key] = {
                        'energy': energy,
                        'diameter': result['channel_diameter'],
                        'distance': distance,
                        'dose_with_channel': result['dose']['value'],
                        'dose_without_channel': ref_dose,
                        'streaming_ratio': ratio
                    }
                    
                    logger.info(f"Streaming ratio for {key}: {ratio:.2f}")
        
        return streaming_ratios

@timeit
def analyze_penetration_laws(results):
    """
    Analyze penetration laws (attenuation, inverse square law deviations).
    
    Parameters:
    -----------
    results : list
        List of simulation result dictionaries
    
    Returns:
    --------
    penetration_laws : dict
        Dictionary of penetration law analysis results
    """
    with LogSection("Analyzing penetration laws"):
        # Group results by energy, diameter, and angle
        grouped_results = {}
        
        for result in results:
            energy = result['energy']
            diameter = result['channel_diameter']
            angle = result['detector_angle']
            
            key = f"E{energy}_D{diameter}_A{angle}"
            if key not in grouped_results:
                grouped_results[key] = []
            
            grouped_results[key].append(result)
        
        penetration_laws = {}
        
        # Analyze distance dependence for each group
        for key, group in grouped_results.items():
            if len(group) < 3:  # Need at least 3 points for meaningful fitting
                continue
            
            # Sort by distance
            group.sort(key=lambda r: r['detector_distance'])
            
            distances = np.array([r['detector_distance'] for r in group])
            doses = np.array([r['dose']['value'] for r in group])
            
            # Extract parameters from key
            parts = key.split('_')
            energy = float(parts[0][1:])
            diameter = float(parts[1][1:])
            angle = float(parts[2][1:])
            
            # Try fitting different models
            
            # 1. Inverse square law: dose = A / r²
            def inverse_square(r, A):
                return A / (r**2)
            
            # 2. Modified inverse square: dose = A / r^n
            def modified_inverse_square(r, A, n):
                return A / (r**n)
            
            # 3. Attenuation with buildup: dose = A * exp(-μ*r) * B / r²
            def attenuation_with_buildup(r, A, mu, B):
                return A * np.exp(-mu * r) * B / (r**2)
            
            # Fit the models
            try:
                # Inverse square
                popt_inv_sq, _ = optimize.curve_fit(inverse_square, distances, doses)
                fitted_inv_sq = inverse_square(distances, *popt_inv_sq)
                
                # Calculate R-squared
                ss_tot_inv_sq = np.sum((doses - np.mean(doses))**2)
                ss_res_inv_sq = np.sum((doses - fitted_inv_sq)**2)
                r_squared_inv_sq = 1 - (ss_res_inv_sq / ss_tot_inv_sq)
                
                # Modified inverse square
                popt_mod_inv_sq, _ = optimize.curve_fit(modified_inverse_square, distances, doses)
                fitted_mod_inv_sq = modified_inverse_square(distances, *popt_mod_inv_sq)
                
                # Calculate R-squared
                ss_tot_mod_inv_sq = np.sum((doses - np.mean(doses))**2)
                ss_res_mod_inv_sq = np.sum((doses - fitted_mod_inv_sq)**2)
                r_squared_mod_inv_sq = 1 - (ss_res_mod_inv_sq / ss_tot_mod_inv_sq)
                
                # Attenuation with buildup
                try:
                    popt_att_buildup, _ = optimize.curve_fit(
                        attenuation_with_buildup, distances, doses,
                        bounds=([0, 0, 0], [np.inf, 1, np.inf])
                    )
                    fitted_att_buildup = attenuation_with_buildup(distances, *popt_att_buildup)
                    
                    # Calculate R-squared
                    ss_tot_att_buildup = np.sum((doses - np.mean(doses))**2)
                    ss_res_att_buildup = np.sum((doses - fitted_att_buildup)**2)
                    r_squared_att_buildup = 1 - (ss_res_att_buildup / ss_tot_att_buildup)
                except:
                    # If attenuation model fails, use simpler models
                    popt_att_buildup = [0, 0, 0]
                    r_squared_att_buildup = 0
                    fitted_att_buildup = np.zeros_like(distances)
                
                # Determine best model
                r_squared_values = [r_squared_inv_sq, r_squared_mod_inv_sq, r_squared_att_buildup]
                best_model_idx = np.argmax(r_squared_values)
                best_model = ['inverse_square', 'modified_inverse_square', 'attenuation_with_buildup'][best_model_idx]
                best_r_squared = r_squared_values[best_model_idx]
                
                penetration_laws[key] = {
                    'energy': energy,
                    'diameter': diameter,
                    'angle': angle,
                    'distances': distances.tolist(),
                    'doses': doses.tolist(),
                    'inverse_square': {
                        'A': popt_inv_sq[0],
                        'r_squared': r_squared_inv_sq
                    },
                    'modified_inverse_square': {
                        'A': popt_mod_inv_sq[0],
                        'n': popt_mod_inv_sq[1],
                        'r_squared': r_squared_mod_inv_sq
                    },
                    'attenuation_with_buildup': {
                        'A': popt_att_buildup[0],
                        'mu': popt_att_buildup[1],
                        'B': popt_att_buildup[2],
                        'r_squared': r_squared_att_buildup
                    },
                    'best_model': best_model,
                    'best_r_squared': best_r_squared
                }
                
                logger.info(f"Penetration law for {key}: Best model = {best_model} (R² = {best_r_squared:.4f})")
                
                if best_model == 'modified_inverse_square':
                    logger.info(f"  Modified inverse square exponent: {popt_mod_inv_sq[1]:.2f}")
                                elif best_model == 'attenuation_with_buildup':
                    logger.info(f"  Attenuation coefficient: {popt_att_buildup[1]:.4f} cm⁻¹")
                
            except Exception as e:
                logger.warning(f"Failed to fit penetration models for {key}: {str(e)}")
        
        # Generate summary statistics for different penetration parameters
        summary = {
            'modified_inverse_square_exponents': {},
            'attenuation_coefficients': {}
        }
        
        # Analyze exponents by energy and diameter
        for energy in SOURCE_ENERGIES:
            summary['modified_inverse_square_exponents'][energy] = {}
            summary['attenuation_coefficients'][energy] = {}
            
            for key, laws in penetration_laws.items():
                if laws['energy'] != energy:
                    continue
                
                diameter = laws['diameter']
                
                # For on-axis points only
                if laws['angle'] < 0.1:
                    if diameter not in summary['modified_inverse_square_exponents'][energy]:
                        summary['modified_inverse_square_exponents'][energy][diameter] = []
                    
                    summary['modified_inverse_square_exponents'][energy][diameter].append(
                        laws['modified_inverse_square']['n']
                    )
                    
                    if diameter not in summary['attenuation_coefficients'][energy]:
                        summary['attenuation_coefficients'][energy][diameter] = []
                    
                    summary['attenuation_coefficients'][energy][diameter].append(
                        laws['attenuation_with_buildup']['mu']
                    )
        
        # Calculate averages
        for energy in summary['modified_inverse_square_exponents']:
            for diameter in summary['modified_inverse_square_exponents'][energy]:
                values = summary['modified_inverse_square_exponents'][energy][diameter]
                if values:
                    summary['modified_inverse_square_exponents'][energy][diameter] = {
                        'mean': np.mean(values),
                        'std': np.std(values) if len(values) > 1 else 0
                    }
            
            for diameter in summary['attenuation_coefficients'][energy]:
                values = summary['attenuation_coefficients'][energy][diameter]
                if values:
                    summary['attenuation_coefficients'][energy][diameter] = {
                        'mean': np.mean(values),
                        'std': np.std(values) if len(values) > 1 else 0
                    }
        
        penetration_laws['summary'] = summary
        
        return penetration_laws

@timeit
def analyze_angular_distribution(results):
    """
    Analyze the angular distribution of radiation streaming through channels.
    
    Parameters:
    -----------
    results : list
        List of simulation result dictionaries
    
    Returns:
    --------
    angular_results : dict
        Dictionary of angular distribution analysis results
    """
    with LogSection("Analyzing angular distribution"):
        # Group results by energy, diameter, and distance
        grouped_results = {}
        
        for result in results:
            energy = result['energy']
            diameter = result['channel_diameter']
            distance = result['detector_distance']
            
            key = f"E{energy}_D{diameter}_d{distance}"
            if key not in grouped_results:
                grouped_results[key] = []
            
            grouped_results[key].append(result)
        
        angular_results = {}
        
        # Analyze each group
        for key, group in grouped_results.items():
            if len(group) < 3:  # Need at least 3 angles for meaningful analysis
                continue
            
            # Sort by angle
            group.sort(key=lambda r: r['detector_angle'])
            
            angles = np.array([r['detector_angle'] for r in group])
            doses = np.array([r['dose']['value'] for r in group])
            
            # Extract parameters from key
            parts = key.split('_')
            energy = float(parts[0][1:])
            diameter = float(parts[1][1:])
            distance = float(parts[2][1:])
            
            # Try fitting different models
            
            # 1. Gaussian: dose = A * exp(-angle²/(2*sigma²))
            def gaussian(angle, A, sigma):
                return A * np.exp(-angle**2 / (2 * sigma**2))
            
            # 2. Cosine law: dose = A * cos^n(angle)
            def cosine_law(angle, A, n):
                # Convert degrees to radians
                angle_rad = np.radians(angle)
                return A * np.cos(angle_rad)**n
            
            # 3. Lorentzian: dose = A / (1 + (angle/gamma)²)
            def lorentzian(angle, A, gamma):
                return A / (1 + (angle/gamma)**2)
            
            try:
                # Gaussian
                popt_gauss, _ = optimize.curve_fit(gaussian, angles, doses, bounds=([0, 0], [np.inf, 90]))
                fitted_gauss = gaussian(angles, *popt_gauss)
                
                # Calculate R-squared
                ss_tot_gauss = np.sum((doses - np.mean(doses))**2)
                ss_res_gauss = np.sum((doses - fitted_gauss)**2)
                r_squared_gauss = 1 - (ss_res_gauss / ss_tot_gauss)
                
                # Cosine law
                popt_cos, _ = optimize.curve_fit(cosine_law, angles, doses, bounds=([0, 0], [np.inf, 100]))
                fitted_cos = cosine_law(angles, *popt_cos)
                
                # Calculate R-squared
                ss_tot_cos = np.sum((doses - np.mean(doses))**2)
                ss_res_cos = np.sum((doses - fitted_cos)**2)
                r_squared_cos = 1 - (ss_res_cos / ss_tot_cos)
                
                # Lorentzian
                popt_lorentz, _ = optimize.curve_fit(lorentzian, angles, doses, bounds=([0, 0], [np.inf, 90]))
                fitted_lorentz = lorentzian(angles, *popt_lorentz)
                
                # Calculate R-squared
                ss_tot_lorentz = np.sum((doses - np.mean(doses))**2)
                ss_res_lorentz = np.sum((doses - fitted_lorentz)**2)
                r_squared_lorentz = 1 - (ss_res_lorentz / ss_tot_lorentz)
                
                # Determine best model
                r_squared_values = [r_squared_gauss, r_squared_cos, r_squared_lorentz]
                best_model_idx = np.argmax(r_squared_values)
                best_model = ['gaussian', 'cosine_law', 'lorentzian'][best_model_idx]
                best_r_squared = r_squared_values[best_model_idx]
                
                # Calculate FWHM for Gaussian
                fwhm_gauss = 2.355 * popt_gauss[1]  # 2.355 = 2*sqrt(2*ln(2))
                
                # Calculate effective beam angle (angle where dose drops to 50% of maximum)
                half_max_dose = np.max(doses) / 2
                
                # Find angle where dose drops below half max
                angle_indices = np.where(doses < half_max_dose)[0]
                if len(angle_indices) > 0:
                    effective_beam_angle = angles[angle_indices[0]]
                else:
                    effective_beam_angle = 90  # Assume wide beam if no dropoff
                
                angular_results[key] = {
                    'energy': energy,
                    'diameter': diameter,
                    'distance': distance,
                    'angles': angles.tolist(),
                    'doses': doses.tolist(),
                    'gaussian': {
                        'A': popt_gauss[0],
                        'sigma': popt_gauss[1],
                        'fwhm': fwhm_gauss,
                        'r_squared': r_squared_gauss
                    },
                    'cosine_law': {
                        'A': popt_cos[0],
                        'n': popt_cos[1],
                        'r_squared': r_squared_cos
                    },
                    'lorentzian': {
                        'A': popt_lorentz[0],
                        'gamma': popt_lorentz[1],
                        'r_squared': r_squared_lorentz
                    },
                    'best_model': best_model,
                    'best_r_squared': best_r_squared,
                    'effective_beam_angle': effective_beam_angle
                }
                
                logger.info(f"Angular distribution for {key}: Best model = {best_model} (R² = {best_r_squared:.4f})")
                
                if best_model == 'gaussian':
                    logger.info(f"  Gaussian FWHM: {fwhm_gauss:.2f}°")
                elif best_model == 'cosine_law':
                    logger.info(f"  Cosine law exponent: {popt_cos[1]:.2f}")
                
                logger.info(f"  Effective beam angle: {effective_beam_angle:.2f}°")
                
            except Exception as e:
                logger.warning(f"Failed to fit angular distribution models for {key}: {str(e)}")
        
        # Generate summary statistics
        summary = {
            'effective_beam_angles': {},
            'angular_width_vs_distance': {},
            'width_vs_diameter': {}
        }
        
        # Group by energy and diameter
        for energy in SOURCE_ENERGIES:
            summary['effective_beam_angles'][energy] = {}
            summary['angular_width_vs_distance'][energy] = {}
            
            # Collect beam angles by diameter
            for key, result in angular_results.items():
                if result['energy'] != energy:
                    continue
                
                diameter = result['diameter']
                distance = result['distance']
                
                if diameter not in summary['effective_beam_angles'][energy]:
                    summary['effective_beam_angles'][energy][diameter] = []
                
                summary['effective_beam_angles'][energy][diameter].append(
                    (distance, result['effective_beam_angle'])
                )
            
            # Calculate average beam angle for each diameter
            for diameter in summary['effective_beam_angles'][energy]:
                points = summary['effective_beam_angles'][energy][diameter]
                points.sort(key=lambda p: p[0])  # Sort by distance
                
                # Group by distance
                distances = []
                angles = []
                
                for dist, angle in points:
                    distances.append(dist)
                    angles.append(angle)
                
                summary['angular_width_vs_distance'][energy][diameter] = {
                    'distances': distances,
                    'angles': angles
                }
        
        # Analyze diameter dependence
        for energy in SOURCE_ENERGIES:
            summary['width_vs_diameter'][energy] = {}
            
            # For fixed distance (30cm), check beam width vs diameter
            diameters = []
            widths = []
            
            for key, result in angular_results.items():
                if result['energy'] != energy or abs(result['distance'] - 30) > 0.1:
                    continue
                
                diameters.append(result['diameter'])
                
                # Use FWHM if Gaussian is best model, otherwise use effective beam angle
                if result['best_model'] == 'gaussian':
                    widths.append(result['gaussian']['fwhm'])
                else:
                    widths.append(result['effective_beam_angle'])
            
            # Sort by diameter
            if diameters and widths:
                sort_indices = np.argsort(diameters)
                diameters = [diameters[i] for i in sort_indices]
                widths = [widths[i] for i in sort_indices]
                
                summary['width_vs_diameter'][energy] = {
                    'diameters': diameters,
                    'widths': widths
                }
        
        angular_results['summary'] = summary
        
        return angular_results

@timeit
def calculate_streaming_path_length(results, wall_thickness=WALL_THICKNESS):
    """
    Calculate the effective path length through the streaming channel.
    
    Parameters:
    -----------
    results : list
        List of simulation result dictionaries
    wall_thickness : float
        Thickness of the shield wall in cm
    
    Returns:
    --------
    path_analysis : dict
        Dictionary of path length analysis results
    """
    with LogSection("Calculating streaming path lengths"):
        # Filter on-axis results
        on_axis_results = [r for r in results if abs(r['detector_angle']) < 0.1]
        
        path_analysis = {}
        
        # Calculate for each energy and diameter
        for energy in set(r['energy'] for r in on_axis_results):
            path_analysis[energy] = {}
            
            energy_results = [r for r in on_axis_results if abs(r['energy'] - energy) < 0.01]
            
            for diameter in set(r['channel_diameter'] for r in energy_results):
                # Skip if diameter is 0 (solid wall)
                if diameter < 0.01:
                    continue
                
                # Get results for this diameter and sort by distance
                diameter_results = [r for r in energy_results 
                                  if abs(r['channel_diameter'] - diameter) < 0.01]
                diameter_results.sort(key=lambda r: r['detector_distance'])
                
                # Calculate ratio of direct path to scattered path contribution
                # For on-axis points, use analytical approximation
                
                # L/D ratio (length to diameter)
                l_d_ratio = wall_thickness / diameter
                
                # Estimate direct fraction using analytical approximation
                # For small L/D, direct fraction is large, for large L/D it's small
                direct_fraction = 1 / (1 + 0.1 * l_d_ratio**2)
                
                # Calculate aspect ratio parameters
                aspect_ratio = wall_thickness / (diameter/2)
                solid_angle = np.pi * (diameter/2)**2 / (wall_thickness**2)
                
                path_analysis[energy][diameter] = {
                    'wall_thickness': wall_thickness,
                    'aspect_ratio': aspect_ratio,
                    'l_d_ratio': l_d_ratio,
                    'solid_angle': solid_angle,
                    'direct_fraction': direct_fraction,
                    'effective_path_length': wall_thickness * (1 - 0.5 * direct_fraction)
                }
                
                logger.info(f"Channel analysis for E={energy} MeV, D={diameter} cm: "
                          f"L/D ratio={l_d_ratio:.2f}, Direct fraction={direct_fraction:.2f}")
        
        return path_analysis

@timeit
def plot_streaming_analysis(results, streaming_analysis):
    """
    Create plots for streaming analysis results.
    
    Parameters:
    -----------
    results : list
        List of simulation result dictionaries
    streaming_analysis : dict
        Dictionary of streaming analysis results
    
        Returns:
    --------
    figs : list
        List of figures created
    """
    with LogSection("Creating streaming analysis plots"):
        # Create output directory
        output_dir = PLOTS_DIR / "streaming_analysis"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        figs = []
        
        # Plot streaming ratios
        if 'streaming_ratios' in streaming_analysis:
            streaming_ratios = streaming_analysis['streaming_ratios']
            
            # Group by energy and distance
            energy_distance_groups = {}
            
            for key, data in streaming_ratios.items():
                energy = data['energy']
                distance = data['distance']
                
                group_key = f"E{energy}_d{distance}"
                if group_key not in energy_distance_groups:
                    energy_distance_groups[group_key] = []
                
                energy_distance_groups[group_key].append(data)
            
            # Plot streaming ratio vs diameter for each energy and distance
            for group_key, group_data in energy_distance_groups.items():
                if len(group_data) < 2:
                    continue
                
                # Sort by diameter
                group_data.sort(key=lambda d: d['diameter'])
                
                diameters = [d['diameter'] for d in group_data]
                ratios = [d['streaming_ratio'] for d in group_data]
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.plot(diameters, ratios, 'o-', linewidth=2, markersize=8)
                
                energy = group_data[0]['energy']
                distance = group_data[0]['distance']
                
                ax.set_title(f'Streaming Ratio vs. Channel Diameter (E={energy} MeV, d={distance} cm)')
                ax.set_xlabel('Channel Diameter (cm)')
                ax.set_ylabel('Dose Ratio (with channel / without channel)')
                ax.set_yscale('log')
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Add best fit line (power law)
                try:
                    # Filter out zeros
                    valid_indices = [i for i, d in enumerate(diameters) if d > 0]
                    valid_diameters = [diameters[i] for i in valid_indices]
                    valid_ratios = [ratios[i] for i in valid_indices]
                    
                    if len(valid_diameters) >= 2:
                        def power_law(x, a, b):
                            return a * x**b
                        
                        popt, _ = optimize.curve_fit(power_law, valid_diameters, valid_ratios)
                        
                        x_fit = np.linspace(min(valid_diameters), max(valid_diameters), 100)
                        y_fit = power_law(x_fit, *popt)
                        
                        ax.plot(x_fit, y_fit, 'r--', linewidth=1.5, 
                               label=f'Fit: {popt[0]:.2e} × D^{popt[1]:.2f}')
                        ax.legend()
                except Exception as e:
                    logger.warning(f"Failed to fit power law for {group_key}: {str(e)}")
                
                # Save figure
                filename = f"streaming_ratio_{group_key}.png"
                plt.tight_layout()
                plt.savefig(output_dir / filename, dpi=300)
                plt.close()
                logger.info(f"Created streaming ratio plot: {filename}")
                
                figs.append(fig)
        
        # Plot angular distributions
        if 'angular_distribution' in streaming_analysis:
            angular_results = streaming_analysis['angular_distribution']
            
            # Filter out summary
            angular_cases = {k: v for k, v in angular_results.items() if k != 'summary'}
            
            for key, data in angular_cases.items():
                if 'angles' not in data or 'doses' not in data:
                    continue
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                angles = data['angles']
                doses = data['doses']
                
                ax.plot(angles, doses, 'ko', markersize=6)
                
                energy = data['energy']
                diameter = data['diameter']
                distance = data['distance']
                
                # Plot best fit model
                best_model = data['best_model']
                angle_fit = np.linspace(0, max(angles), 100)
                
                if best_model == 'gaussian':
                    params = data['gaussian']
                    dose_fit = params['A'] * np.exp(-angle_fit**2 / (2 * params['sigma']**2))
                    model_label = f'Gaussian (σ={params["sigma"]:.2f}°, FWHM={params["fwhm"]:.2f}°)'
                elif best_model == 'cosine_law':
                    params = data['cosine_law']
                    angle_rad = np.radians(angle_fit)
                    dose_fit = params['A'] * np.cos(angle_rad)**params['n']
                    model_label = f'Cosine^{params["n"]:.2f}'
                elif best_model == 'lorentzian':
                    params = data['lorentzian']
                    dose_fit = params['A'] / (1 + (angle_fit/params['gamma'])**2)
                    model_label = f'Lorentzian (γ={params["gamma"]:.2f}°)'
                
                ax.plot(angle_fit, dose_fit, 'r-', linewidth=2, label=model_label)
                
                # Mark effective beam angle
                if 'effective_beam_angle' in data:
                    beam_angle = data['effective_beam_angle']
                    ax.axvline(beam_angle, color='blue', linestyle='--', 
                              label=f'Beam angle={beam_angle:.1f}°')
                
                ax.set_title(f'Angular Dose Distribution (E={energy} MeV, D={diameter} cm, d={distance} cm)')
                ax.set_xlabel('Angle (degrees)')
                ax.set_ylabel('Dose')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                
                # Save figure
                filename = f"angular_distribution_{key}.png"
                plt.tight_layout()
                plt.savefig(output_dir / filename, dpi=300)
                plt.close()
                logger.info(f"Created angular distribution plot: {filename}")
                
                figs.append(fig)
            
            # Plot beam width vs diameter for different energies
            if 'summary' in angular_results and 'width_vs_diameter' in angular_results['summary']:
                width_data = angular_results['summary']['width_vs_diameter']
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for energy, data in width_data.items():
                    if 'diameters' in data and 'widths' in data and len(data['diameters']) > 1:
                        ax.plot(data['diameters'], data['widths'], 'o-', 
                               label=f'E = {energy} MeV')
                
                ax.set_title('Beam Width vs Channel Diameter')
                ax.set_xlabel('Channel Diameter (cm)')
                ax.set_ylabel('Beam Width (degrees)')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                
                # Save figure
                filename = "beam_width_vs_diameter.png"
                plt.tight_layout()
                plt.savefig(output_dir / filename, dpi=300)
                plt.close()
                logger.info(f"Created beam width plot: {filename}")
                
                figs.append(fig)
        
        # Plot penetration laws
        if 'penetration_laws' in streaming_analysis:
            penetration = streaming_analysis['penetration_laws']
            
            # Filter out summary
            penetration_cases = {k: v for k, v in penetration.items() if k != 'summary'}
            
            for key, data in penetration_cases.items():
                if 'distances' not in data or 'doses' not in data:
                    continue
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                distances = data['distances']
                doses = data['doses']
                
                ax.plot(distances, doses, 'ko', markersize=6)
                
                energy = data['energy']
                diameter = data['diameter']
                angle = data['angle']
                
                # Plot best fit model
                best_model = data.get('best_model')
                if best_model:
                    dist_fit = np.linspace(min(distances), max(distances), 100)
                    
                    if best_model == 'inverse_square':
                        params = data['inverse_square']
                        dose_fit = params['A'] / (dist_fit**2)
                        model_label = f'Inverse Square (1/r²)'
                    elif best_model == 'modified_inverse_square':
                        params = data['modified_inverse_square']
                        dose_fit = params['A'] / (dist_fit**params['n'])
                        model_label = f'Modified (1/r^{params["n"]:.2f})'
                    elif best_model == 'attenuation_with_buildup':
                        params = data['attenuation_with_buildup']
                        dose_fit = params['A'] * np.exp(-params['mu'] * dist_fit) * params['B'] / (dist_fit**2)
                        model_label = f'Att. + Buildup (μ={params["mu"]:.3f} cm⁻¹)'
                    
                    ax.plot(dist_fit, dose_fit, 'r-', linewidth=2, label=model_label)
                
                ax.set_title(f'Dose vs Distance (E={energy} MeV, D={diameter} cm, θ={angle}°)')
                ax.set_xlabel('Distance (cm)')
                ax.set_ylabel('Dose')
                ax.set_yscale('log')
                ax.set_xscale('log')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                
                # Save figure
                filename = f"penetration_{key}.png"
                plt.tight_layout()
                plt.savefig(output_dir / filename, dpi=300)
                plt.close()
                logger.info(f"Created penetration plot: {filename}")
                
                figs.append(fig)
            
            # Plot inverse square exponents by energy and diameter
            if 'summary' in penetration and 'modified_inverse_square_exponents' in penetration['summary']:
                exponent_data = penetration['summary']['modified_inverse_square_exponents']
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                energies = sorted(exponent_data.keys())
                
                for energy in energies:
                    diameters = sorted(exponent_data[energy].keys())
                    exponents = [exponent_data[energy][d]['mean'] for d in diameters]
                    errors = [exponent_data[energy][d]['std'] for d in diameters]
                    
                    if len(diameters) > 1:
                        ax.errorbar(diameters, exponents, yerr=errors, fmt='o-', 
                                   capsize=5, label=f'E = {energy} MeV')
                
                ax.axhline(2.0, color='black', linestyle='--', alpha=0.5, 
                          label='Inverse Square (n=2)')
                
                ax.set_title('Distance Power Law Exponent vs Channel Diameter')
                ax.set_xlabel('Channel Diameter (cm)')
                ax.set_ylabel('Power Law Exponent (n)')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                
                # Save figure
                filename = "inverse_square_exponents.png"
                plt.tight_layout()
                plt.savefig(output_dir / filename, dpi=300)
                plt.close()
                logger.info(f"Created exponent plot: {filename}")
                
                figs.append(fig)
        
        return figs

@timeit
def perform_streaming_analysis(results):
    """
    Perform comprehensive streaming analysis on simulation results.
    
    Parameters:
    -----------
    results : list
        List of simulation result dictionaries
    
    Returns:
    --------
    analysis_results : dict
        Dictionary of streaming analysis results
    """
    with LogSection("Performing comprehensive streaming analysis"):
        # Calculate streaming ratios
        streaming_ratios = calculate_streaming_ratio(results)
        
        # Analyze penetration laws
        penetration_laws = analyze_penetration_laws(results)
        
        # Analyze angular distributions
        angular_distribution = analyze_angular_distribution(results)
        
        # Calculate streaming path lengths
        path_analysis = calculate_streaming_path_length(results)
        
        # Compile all results
        analysis_results = {
            'streaming_ratios': streaming_ratios,
            'penetration_laws': penetration_laws,
            'angular_distribution': angular_distribution,
            'path_analysis': path_analysis
        }
        
        # Create visualizations
        analysis_results['figures'] = plot_streaming_analysis(results, analysis_results)
        
        return analysis_results


