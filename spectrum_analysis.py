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
        energy_midpoints = spectrum_data['energy_midpoints']
        flux_spectrum = spectrum_data['flux_spectrum']
        
        # Ensure no negative or zero values
        flux_spectrum = np.maximum(flux_spectrum, 1e-30)
        
        # Calculate total flux (integral of spectrum)
        total_flux = np.trapz(flux_spectrum, energy_midpoints)
        
        # Create interpolation function
        interp_spectrum = interpolate.interp1d(
            energy_midpoints, flux_spectrum, kind='linear', 
            bounds_error=False, fill_value=1e-30
        )
        
        # Calculate average energy
        avg_energy_num = np.trapz(energy_midpoints * flux_spectrum, energy_midpoints)
        avg_energy = avg_energy_num / total_flux if total_flux > 0 else 0
        
        # Calculate median energy (energy where cumulative spectrum = 50%)
        cumulative_spectrum = np.cumsum(flux_spectrum) / np.sum(flux_spectrum)
        median_idx = np.searchsorted(cumulative_spectrum, 0.5)
        median_energy = energy_midpoints[median_idx] if median_idx < len(energy_midpoints) else energy_midpoints[-1]
        
        # Calculate FWHM (Full Width at Half Maximum)
        max_flux = np.max(flux_spectrum)
        half_max = max_flux / 2
        max_energy_idx = np.argmax(flux_spectrum)
        max_energy = energy_midpoints[max_energy_idx]
        
        # Find left bound
        left_idx = np.searchsorted((flux_spectrum[:max_energy_idx] - half_max) >= 0, True)
        left_energy = energy_midpoints[left_idx] if left_idx < len(energy_midpoints) else energy_midpoints[0]
        
        # Find right bound
        right_indices = max_energy_idx + np.where((flux_spectrum[max_energy_idx:] - half_max) <= 0)[0]
        right_idx = right_indices[0] if len(right_indices) > 0 else len(energy_midpoints) - 1
        right_energy = energy_midpoints[right_idx] if right_idx < len(energy_midpoints) else energy_midpoints[-1]
        
        fwhm = right_energy - left_energy
        
        # Calculate dose from spectrum
        dose_factors = np.array([flux_to_dose_conversion(e) for e in energy_midpoints])
        dose_spectrum = flux_spectrum * dose_factors
        total_dose = np.trapz(dose_spectrum, energy_midpoints)
        
        # Calculate hardening ratio (ratio of flux above 1 MeV to total flux)
        high_energy_indices = energy_midpoints >= 1.0
        if np.any(high_energy_indices):
            high_energy_flux = np.trapz(
                flux_spectrum[high_energy_indices], 
                energy_midpoints[high_energy_indices]
            )
            hardening_ratio = high_energy_flux / total_flux if total_flux > 0 else 0
        else:
            hardening_ratio = 0
        
        # Collect metrics
        metrics = {
            'total_flux': total_flux,
            'average_energy': avg_energy,
            'median_energy': median_energy,
            'peak_energy': max_energy,
            'fwhm': fwhm,
            'total_dose': total_dose,
            'hardening_ratio': hardening_ratio,
            'dose_per_flux': total_dose / total_flux if total_flux > 0 else 0
        }
        
        logger.info(f"Spectrum metrics: Average energy = {avg_energy:.3f} MeV, "
                  f"Median energy = {median_energy:.3f} MeV, FWHM = {fwhm:.3f} MeV")
        
        return metrics

@timeit
def analyze_buildup_factor(results):
    """
    Analyze buildup factors from spectrum results.
    
    Parameters:
    -----------
    results : list
        List of simulation result dictionaries
    
    Returns:
    --------
    buildup_data : dict
        Dictionary of buildup factor analysis results
    """
    with LogSection("Analyzing buildup factors"):
        # Filter results with spectrum data
        spectrum_results = [r for r in results if 'energy_spectrum' in r]
        
        if not spectrum_results:
            logger.warning("No spectrum data available for buildup analysis")
            return {}
        
        # Group by energy, diameter, and distance (only on-axis points)
        on_axis_results = [r for r in spectrum_results if abs(r['detector_angle']) < 0.1]
        
        buildup_data = {}
        
        for result in on_axis_results:
            energy = result['energy']
            diameter = result['channel_diameter']
            distance = result['detector_distance']
            
            key = f"E{energy}_D{diameter}_d{distance}"
            
            # Calculate uncollided flux
            spectrum = result['energy_spectrum']
            uncollided_idx = np.abs(np.array(spectrum['energy_midpoints']) - energy).argmin()
            uncollided_flux = spectrum['flux_spectrum'][uncollided_idx]
            
            # Get total flux
            total_flux = np.sum(spectrum['flux_spectrum'])
            
            # Calculate buildup factor
            buildup_factor = total_flux / uncollided_flux if uncollided_flux > 0 else np.nan
            
            # Store in buildup data
            buildup_data[key] = {
                'energy': energy,
                'diameter': diameter,
                'distance': distance,
                'uncollided_flux': uncollided_flux,
                'total_flux': total_flux,
                'buildup_factor': buildup_factor
            }
            
            logger.info(f"Buildup factor for {key}: {buildup_factor:.2f}")
        
        # Create summaries by energy
        energy_summary = {}
        for energy in set(r['energy'] for r in on_axis_results):
            energy_keys = [k for k in buildup_data.keys() if f"E{energy}_" in k]
            buildup_values = [buildup_data[k]['buildup_factor'] for k in energy_keys 
                            if not np.isnan(buildup_data[k]['buildup_factor'])]
            
            if buildup_values:
                energy_summary[energy] = {
                    'mean': np.mean(buildup_values),
                    'min': np.min(buildup_values),
                    'max': np.max(buildup_values),
                    'std': np.std(buildup_values)
                }
        
        buildup_data['energy_summary'] = energy_summary
        
        return buildup_data

@timeit
def analyze_spectral_changes(results):
    """
    Analyze how the energy spectrum changes with distance, angle, and channel size.
    
    Parameters:
    -----------
    results : list
        List of simulation result dictionaries
    
    Returns:
    --------
    spectral_changes : dict
        Dictionary of spectral change analysis results
    """
    with LogSection("Analyzing spectral changes"):
        # Filter results with spectrum data
        spectrum_results = [r for r in results if 'energy_spectrum' in r]
        
        if not spectrum_results:
            logger.warning("No spectrum data available for spectral change analysis")
            return {}
        
        spectral_changes = {
            'distance_effects': {},
            'angle_effects': {},
            'diameter_effects': {}
        }
        
        # Analyze distance effects (for fixed energy, diameter, and angle=0)
        for energy in set(r['energy'] for r in spectrum_results):
            for diameter in set(r['channel_diameter'] for r in spectrum_results):
                # Get on-axis results for this energy and diameter
                on_axis = [r for r in spectrum_results 
                         if abs(r['energy'] - energy) < 0.01 
                         and abs(r['channel_diameter'] - diameter) < 0.01
                         and abs(r['detector_angle']) < 0.1]
                
                if len(on_axis) < 2:  # Need at least 2 distances to compare
                    continue
                
                # Sort by distance
                on_axis.sort(key=lambda r: r['detector_distance'])
                
                # Calculate spectrum metrics for each distance
                metrics_by_distance = {}
                for r in on_axis:
                    metrics = calculate_spectrum_metrics(r['energy_spectrum'])
                    metrics_by_distance[r['detector_distance']] = metrics
                
                # Calculate changes with distance
                distances = sorted(metrics_by_distance.keys())
                ref_metrics = metrics_by_distance[distances[0]]  # Reference is closest distance
                
                distance_changes = {
                    'energy': energy,
                    'diameter': diameter,
                    'reference_distance': distances[0],
                    'distances': distances,
                    'average_energy_change': [],
                    'hardening_ratio_change': [],
                    'dose_per_flux_change': []
                }
                
                for dist in distances[1:]:
                    curr_metrics = metrics_by_distance[dist]
                    
                    # Calculate relative changes
                    avg_energy_change = (curr_metrics['average_energy'] / ref_metrics['average_energy'] - 1) * 100
                    hardening_change = (curr_metrics['hardening_ratio'] / ref_metrics['hardening_ratio'] - 1) * 100 if ref_metrics['hardening_ratio'] > 0 else 0
                    dose_per_flux_change = (curr_metrics['dose_per_flux'] / ref_metrics['dose_per_flux'] - 1) * 100 if ref_metrics['dose_per_flux'] > 0 else 0
                    
                    distance_changes['average_energy_change'].append(avg_energy_change)
                    distance_changes['hardening_ratio_change'].append(hardening_change)
                    distance_changes['dose_per_flux_change'].append(dose_per_flux_change)
                
                key = f"E{energy}_D{diameter}"
                spectral_changes['distance_effects'][key] = distance_changes
                
                logger.info(f"Spectral changes with distance for {key}: "
                          f"Avg energy change: {distance_changes['average_energy_change'][-1]:.1f}%, "
                          f"Hardening change: {distance_changes['hardening_ratio_change'][-1]:.1f}%")
        
        # Analyze angle effects (for fixed energy, diameter, and distance)
        for energy in set(r['energy'] for r in spectrum_results):
            for diameter in set(r['channel_diameter'] for r in spectrum_results):
                for distance in set(r['detector_distance'] for r in spectrum_results):
                    # Get results for this energy, diameter, and distance
                    angle_results = [r for r in spectrum_results 
                                   if abs(r['energy'] - energy) < 0.01 
                                   and abs(r['channel_diameter'] - diameter) < 0.01
                                   and abs(r['detector_distance'] - distance) < 0.1]
                    
                    if len(angle_results) < 2:  # Need at least 2 angles to compare
                        continue
                    
                    # Sort by angle
                    angle_results.sort(key=lambda r: r['detector_angle'])
                    
                    # Calculate spectrum metrics for each angle
                    metrics_by_angle = {}
                    for r in angle_results:
                        metrics = calculate_spectrum_metrics(r['energy_spectrum'])
                        metrics_by_angle[r['detector_angle']] = metrics
                    
                    # Calculate changes with angle
                    angles = sorted(metrics_by_angle.keys())
                    ref_metrics = metrics_by_angle[angles[0]]  # Reference is 0 degrees
                    
                    angle_changes = {
                        'energy': energy,
                        'diameter': diameter,
                        'distance': distance,
                        'reference_angle': angles[0],
                        'angles': angles,
                        'average_energy_change': [],
                        'hardening_ratio_change': [],
                        'dose_per_flux_change': []
                    }
                    
                    for angle in angles[1:]:
                        curr_metrics = metrics_by_angle[angle]
                        
                        # Calculate relative changes
                        avg_energy_change = (curr_metrics['average_energy'] / ref_metrics['average_energy'] - 1) * 100
                        hardening_change = (curr_metrics['hardening_ratio'] / ref_metrics['hardening_ratio'] - 1) * 100 if ref_metrics['hardening_ratio'] > 0 else 0
                        dose_per_flux_change = (curr_metrics['dose_per_flux'] / ref_metrics['dose_per_flux'] - 1) * 100 if ref_metrics['dose_per_flux'] > 0 else 0
                        
                        angle_changes['average_energy_change'].append(avg_energy_change)
                        angle_changes['hardening_ratio_change'].append(hardening_change)
                        angle_changes['dose_per_flux_change'].append(dose_per_flux_change)
                    
                    key = f"E{energy}_D{diameter}_d{distance}"
                    spectral_changes['angle_effects'][key] = angle_changes
                    
                    if len(angles) > 1:
                        logger.info(f"Spectral changes with angle for {key}: "
                                  f"Avg energy change at {angles[-1]}°: {angle_changes['average_energy_change'][-1]:.1f}%, "
                                  f"Hardening change: {angle_changes['hardening_ratio_change'][-1]:.1f}%")
        
        # Generate summary of spectral changes
        spectral_changes['summary'] = {
            'distance_hardening': {},
            'angle_softening': {}
        }
        
        # Summarize distance hardening by energy
        for energy in set(r['energy'] for r in spectrum_results):
            energy_keys = [k for k in spectral_changes['distance_effects'].keys() if f"E{energy}_"
                        energy_keys = [k for k in spectral_changes['distance_effects'].keys() if f"E{energy}_" in k]
            hardening_values = []
            
            for key in energy_keys:
                if 'hardening_ratio_change' in spectral_changes['distance_effects'][key]:
                    changes = spectral_changes['distance_effects'][key]['hardening_ratio_change']
                    if changes:  # If not empty
                        hardening_values.append(changes[-1])  # Use the last value (largest distance)
            
            if hardening_values:
                spectral_changes['summary']['distance_hardening'][energy] = {
                    'mean': np.mean(hardening_values),
                    'min': np.min(hardening_values),
                    'max': np.max(hardening_values)
                }
        
        # Summarize angle softening by energy
        for energy in set(r['energy'] for r in spectrum_results):
            energy_keys = [k for k in spectral_changes['angle_effects'].keys() if f"E{energy}_" in k]
            softening_values = []
            
            for key in energy_keys:
                if 'average_energy_change' in spectral_changes['angle_effects'][key]:
                    changes = spectral_changes['angle_effects'][key]['average_energy_change']
                    if changes:  # If not empty
                        softening_values.append(changes[-1])  # Use the last value (largest angle)
            
            if softening_values:
                spectral_changes['summary']['angle_softening'][energy] = {
                    'mean': np.mean(softening_values),
                    'min': np.min(softening_values),
                    'max': np.max(softening_values)
                }
        
        return spectral_changes

@timeit
def fit_analytical_spectrum(spectrum_data, source_energy):
    """
    Fit an analytical model to the observed energy spectrum.
    
    Parameters:
    -----------
    spectrum_data : dict
        Dictionary containing 'energy_midpoints' and 'flux_spectrum'
    source_energy : float
        Original source energy in MeV
    
    Returns:
    --------
    fit_results : dict
        Dictionary of fit parameters and metrics
    """
    with LogSection(f"Fitting analytical spectrum model (source: {source_energy} MeV)"):
        # Extract data
        energy_midpoints = spectrum_data['energy_midpoints']
        flux_spectrum = spectrum_data['flux_spectrum']
        
        # Normalize the spectrum
        flux_spectrum_norm = flux_spectrum / np.max(flux_spectrum)
        
        # Define analytical models
        
        # 1. Klein-Nishina + exponential model for Compton-scattered spectrum
        def klein_nishina_model(energy, A, mu, sigma_uncollided):
            # Uncollided peak (Gaussian)
            uncollided = A * np.exp(-0.5 * ((energy - source_energy) / sigma_uncollided)**2)
            
            # Klein-Nishina shape for Compton scattered spectrum
            alpha = source_energy / 0.511  # source energy in units of electron rest energy
            
            # Only calculate for energies less than the Compton edge
            compton_edge = source_energy / (1 + 2*alpha)
            
            # Initialize scattered component
            scattered = np.zeros_like(energy)
            
            # For energies below Compton edge
            mask = energy < compton_edge
            if np.any(mask):
                e_ratio = energy[mask] / source_energy
                kn_factor = 1 / e_ratio + e_ratio - np.sin(np.arccos(1 - (1 - e_ratio) / alpha))**2
                
                # Apply attenuation and normalization
                scattered[mask] = (1 - A) * kn_factor * np.exp(-mu * (source_energy - energy[mask]))
            
            return uncollided + scattered
        
        # 2. Simple multi-component model with exponential tail
        def multi_component_model(energy, A_peak, sigma_peak, A_scatter, mu_scatter, A_bkg):
            # Uncollided peak
            peak = A_peak * np.exp(-0.5 * ((energy - source_energy) / sigma_peak)**2)
            
            # Scattered component with exponential
            scatter = A_scatter * np.exp(-mu_scatter * (source_energy - energy))
            
            # Flat background/low-energy component
            background = A_bkg * np.ones_like(energy)
            
            # Only apply scatter term below source energy
            scatter[energy > source_energy] = 0
            
            return peak + scatter + background
        
        # Try fitting the multi-component model first (usually more robust)
        try:
            # Initial parameter guesses
            p0_multi = [
                0.5,           # A_peak: relative amplitude of peak
                0.05,          # sigma_peak: width of peak
                0.4,           # A_scatter: relative amplitude of scatter
                2.0,           # mu_scatter: exponential slope
                0.1            # A_bkg: background level
            ]
            
            # Bounds for parameters
            bounds_multi = ([0, 0.001, 0, 0.1, 0], [1, 0.2, 1, 10, 0.5])
            
            # Fit the model
            popt_multi, pcov_multi = optimize.curve_fit(
                multi_component_model, energy_midpoints, flux_spectrum_norm,
                p0=p0_multi, bounds=bounds_multi, maxfev=10000
            )
            
            # Calculate fitted values
            fit_multi = multi_component_model(energy_midpoints, *popt_multi)
            
            # Calculate R-squared
            ss_tot = np.sum((flux_spectrum_norm - np.mean(flux_spectrum_norm))**2)
            ss_res = np.sum((flux_spectrum_norm - fit_multi)**2)
            r_squared_multi = 1 - (ss_res / ss_tot)
            
            # Calculate parameter errors
            perr_multi = np.sqrt(np.diag(pcov_multi))
            
            multi_fit_results = {
                'model': 'multi_component',
                'parameters': {
                    'A_peak': popt_multi[0],
                    'sigma_peak': popt_multi[1],
                    'A_scatter': popt_multi[2],
                    'mu_scatter': popt_multi[3],
                    'A_bkg': popt_multi[4]
                },
                'parameter_errors': {
                    'A_peak_err': perr_multi[0],
                    'sigma_peak_err': perr_multi[1],
                    'A_scatter_err': perr_multi[2],
                    'mu_scatter_err': perr_multi[3],
                    'A_bkg_err': perr_multi[4]
                },
                'r_squared': r_squared_multi,
                'fitted_values': fit_multi
            }
            
            logger.info(f"Multi-component fit: R² = {r_squared_multi:.4f}, "
                      f"Peak fraction = {popt_multi[0]:.4f}, "
                      f"Scatter attenuation = {popt_multi[3]:.4f}")
            
        except Exception as e:
            logger.warning(f"Multi-component model fitting failed: {str(e)}")
            multi_fit_results = {'model': 'multi_component', 'error': str(e)}
        
        # Try Klein-Nishina model also
        try:
            # Initial parameter guesses
            p0_kn = [
                0.3,    # A: amplitude of uncollided peak
                3.0,    # mu: attenuation coefficient
                0.05    # sigma_uncollided: width of uncollided peak
            ]
            
            # Bounds for parameters
            bounds_kn = ([0, 0.1, 0.001], [1, 10, 0.2])
            
            # Fit the model
            popt_kn, pcov_kn = optimize.curve_fit(
                klein_nishina_model, energy_midpoints, flux_spectrum_norm,
                p0=p0_kn, bounds=bounds_kn, maxfev=10000
            )
            
            # Calculate fitted values
            fit_kn = klein_nishina_model(energy_midpoints, *popt_kn)
            
            # Calculate R-squared
            ss_tot = np.sum((flux_spectrum_norm - np.mean(flux_spectrum_norm))**2)
            ss_res = np.sum((flux_spectrum_norm - fit_kn)**2)
            r_squared_kn = 1 - (ss_res / ss_tot)
            
            # Calculate parameter errors
            perr_kn = np.sqrt(np.diag(pcov_kn))
            
            kn_fit_results = {
                'model': 'klein_nishina',
                'parameters': {
                    'A': popt_kn[0],
                    'mu': popt_kn[1],
                    'sigma_uncollided': popt_kn[2]
                },
                'parameter_errors': {
                    'A_err': perr_kn[0],
                    'mu_err': perr_kn[1],
                    'sigma_uncollided_err': perr_kn[2]
                },
                'r_squared': r_squared_kn,
                'fitted_values': fit_kn
            }
            
            logger.info(f"Klein-Nishina fit: R² = {r_squared_kn:.4f}, "
                      f"Uncollided fraction = {popt_kn[0]:.4f}, "
                      f"Attenuation = {popt_kn[1]:.4f}")
            
        except Exception as e:
            logger.warning(f"Klein-Nishina model fitting failed: {str(e)}")
            kn_fit_results = {'model': 'klein_nishina', 'error': str(e)}
        
        # Select best model based on R-squared
        if ('r_squared' in multi_fit_results and 'r_squared' in kn_fit_results):
            if multi_fit_results['r_squared'] > kn_fit_results['r_squared']:
                best_model = 'multi_component'
                best_r_squared = multi_fit_results['r_squared']
            else:
                best_model = 'klein_nishina'
                best_r_squared = kn_fit_results['r_squared']
            
            logger.info(f"Best fit model: {best_model} (R² = {best_r_squared:.4f})")
        elif 'r_squared' in multi_fit_results:
            best_model = 'multi_component'
            best_r_squared = multi_fit_results['r_squared']
        elif 'r_squared' in kn_fit_results:
            best_model = 'klein_nishina'
            best_r_squared = kn_fit_results['r_squared']
        else:
            best_model = None
            best_r_squared = None
            logger.warning("Both model fits failed")
        
        # Return both models and the best one
        return {
            'multi_component': multi_fit_results,
            'klein_nishina': kn_fit_results,
            'best_model': best_model,
            'best_r_squared': best_r_squared,
            'energy_midpoints': energy_midpoints,
            'flux_spectrum_norm': flux_spectrum_norm,
            'source_energy': source_energy
        }

@timeit
def plot_spectrum_analysis(spectrum_results, analysis_results):
    """
    Create plots for spectrum analysis results.
    
    Parameters:
    -----------
    spectrum_results : list
        List of results containing spectrum data
    analysis_results : dict
        Dictionary of spectrum analysis results
    
    Returns:
    --------
    figs : list
        List of figures created
    """
    with LogSection("Creating spectrum analysis plots"):
        # Create output directory
        output_dir = PLOTS_DIR / "spectrum_analysis"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        figs = []
        
        # Plot buildup factors
        if 'energy_summary' in analysis_results.get('buildup', {}):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            energy_summary = analysis_results['buildup']['energy_summary']
            energies = sorted(energy_summary.keys())
            
            means = [energy_summary[e]['mean'] for e in energies]
            mins = [energy_summary[e]['min'] for e in energies]
            maxs = [energy_summary[e]['max'] for e in energies]
            
            ax.errorbar(energies, means, yerr=[np.array(means)-np.array(mins), 
                                             np.array(maxs)-np.array(means)],
                      fmt='o-', capsize=5, elinewidth=1, markeredgewidth=1)
            
            ax.set_title('Energy-Dependent Buildup Factor')
            ax.set_xlabel('Source Energy (MeV)')
            ax.set_ylabel('Buildup Factor')
            ax.set_ylim(bottom=0)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Save figure
            filename = "buildup_factors.png"
            plt.tight_layout()
            plt.savefig(output_dir / filename, dpi=300)
            plt.close()
            logger.info(f"Created buildup factor plot: {filename}")
            
            figs.append(fig)
        
        # Plot spectral changes with distance
        if analysis_results.get('spectral_changes', {}).get('distance_effects'):
            distance_effects = analysis_results['spectral_changes']['distance_effects']
            
            # Group by energy
            energies = set()
            for key in distance_effects.keys():
                if key.startswith('E'):
                    energy = float(key.split('_')[0][1:])
                    energies.add(energy)
            
            for energy in sorted(energies):
                energy_keys = [k for k in distance_effects.keys() if k.startswith(f'E{energy}_')]
                
                if not energy_keys:
                    continue
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for key in energy_keys:
                    data = distance_effects[key]
                    diameter = data['diameter']
                    distances = data['distances'][1:]  # Skip reference distance
                    
                    if 'average_energy_change' in data and data['average_energy_change']:
                        ax.plot(distances, data['average_energy_change'], 'o-', 
                               label=f'Ø = {diameter:.2f} cm')
                
                ax.set_title(f'Spectral Hardening with Distance (E = {energy} MeV)')
                ax.set_xlabel('Detector Distance (cm)')
                ax.set_ylabel('Average Energy Change (%)')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                
                # Save figure
                                # Save figure
                filename = f"spectral_hardening_E{energy}.png"
                plt.tight_layout()
                plt.savefig(output_dir / filename, dpi=300)
                plt.close()
                logger.info(f"Created spectral hardening plot: {filename}")
                
                figs.append(fig)
        
        # Plot analytical model fits
        if 'spectrum_fits' in analysis_results:
            for case_id, fit_result in analysis_results['spectrum_fits'].items():
                if 'best_model' not in fit_result or fit_result['best_model'] is None:
                    continue
                
                energy_midpoints = fit_result['energy_midpoints']
                flux_spectrum_norm = fit_result['flux_spectrum_norm']
                source_energy = fit_result['source_energy']
                best_model = fit_result['best_model']
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot original spectrum
                ax.plot(energy_midpoints, flux_spectrum_norm, 'ko', markersize=4, alpha=0.5, label='Data')
                
                # Plot fitted model
                best_fit = fit_result[best_model]['fitted_values']
                ax.plot(energy_midpoints, best_fit, 'r-', linewidth=2, label=f'Fit ({best_model})')
                
                # If multi-component model is available, plot components
                if best_model == 'multi_component' and 'parameters' in fit_result['multi_component']:
                    params = fit_result['multi_component']['parameters']
                    
                    # Plot peak component
                    peak = params['A_peak'] * np.exp(-0.5 * ((energy_midpoints - source_energy) / params['sigma_peak'])**2)
                    ax.plot(energy_midpoints, peak, 'g--', alpha=0.7, label='Peak Component')
                    
                    # Plot scatter component
                    scatter = params['A_scatter'] * np.exp(-params['mu_scatter'] * (source_energy - energy_midpoints))
                    scatter[energy_midpoints > source_energy] = 0
                    ax.plot(energy_midpoints, scatter, 'b--', alpha=0.7, label='Scatter Component')
                    
                    # Plot background component
                    background = params['A_bkg'] * np.ones_like(energy_midpoints)
                    ax.plot(energy_midpoints, background, 'm--', alpha=0.7, label='Background Component')
                
                ax.set_title(f'Spectrum Fit (Source E = {source_energy} MeV)')
                ax.set_xlabel('Energy (MeV)')
                ax.set_ylabel('Normalized Flux')
                ax.set_yscale('log')
                ax.set_ylim(bottom=1e-3)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                
                # Save figure
                filename = f"spectrum_fit_{case_id}.png"
                plt.tight_layout()
                plt.savefig(output_dir / filename, dpi=300)
                plt.close()
                logger.info(f"Created spectrum fit plot: {filename}")
                
                figs.append(fig)
        
        return figs

@timeit
def perform_spectrum_analysis(results):
    """
    Perform comprehensive spectrum analysis on simulation results.
    
    Parameters:
    -----------
    results : list
        List of simulation result dictionaries
    
    Returns:
    --------
    analysis_results : dict
        Dictionary of spectrum analysis results
    """
    with LogSection("Performing comprehensive spectrum analysis"):
        # Filter results with spectrum data
        spectrum_results = [r for r in results if 'energy_spectrum' in r]
        
        if not spectrum_results:
            logger.warning("No spectrum data available for analysis")
            return {}
        
        # Calculate metrics for all spectra
        logger.info(f"Calculating metrics for {len(spectrum_results)} spectra")
        for result in spectrum_results:
            result['spectrum_metrics'] = calculate_spectrum_metrics(result['energy_spectrum'])
        
        # Analyze buildup factors
        buildup_results = analyze_buildup_factor(results)
        
        # Analyze spectral changes with distance, angle, etc.
        spectral_changes = analyze_spectral_changes(results)
        
        # Fit analytical models to selected spectra
        spectrum_fits = {}
        
        # Select representative spectra for fitting
        # (on-axis points at 30cm distance for different energies and diameters)
        for energy in SOURCE_ENERGIES:
            filtered_results = [r for r in spectrum_results 
                              if abs(r['energy'] - energy) < 0.01 
                              and abs(r['detector_distance'] - 30) < 0.1
                              and abs(r['detector_angle']) < 0.1]
            
            # Try different diameters
            for result in filtered_results:
                diameter = result['channel_diameter']
                case_id = f"E{energy}_D{diameter}_d30"
                
                spectrum_fits[case_id] = fit_analytical_spectrum(
                    result['energy_spectrum'], energy
                )
        
        # Compile all results
        analysis_results = {
            'metrics': {r.get('id', f"result_{i}"): r['spectrum_metrics'] 
                       for i, r in enumerate(spectrum_results) if 'spectrum_metrics' in r},
            'buildup': buildup_results,
            'spectral_changes': spectral_changes,
            'spectrum_fits': spectrum_fits
        }
        
        # Create visualizations
        analysis_results['figures'] = plot_spectrum_analysis(spectrum_results, analysis_results)
        
        return analysis_results
