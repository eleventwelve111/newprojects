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
        
        # Validate input data
        if len(energy_midpoints) != len(flux_spectrum):
            raise ValueError("Energy midpoints and flux spectrum must have the same length")
        
        if len(energy_midpoints) < 2:
            raise ValueError("Spectrum must have at least two data points")
        
        # Calculate energy bin widths
        if 'energy_bins' in spectrum_data:
            energy_bins = spectrum_data['energy_bins']
            bin_widths = np.diff(energy_bins)
        else:
            # Estimate bin widths from midpoints
            extended_midpoints = np.concatenate([
                [energy_midpoints[0] - (energy_midpoints[1] - energy_midpoints[0])/2],
                energy_midpoints,
                [energy_midpoints[-1] + (energy_midpoints[-1] - energy_midpoints[-2])/2]
            ])
            bin_widths = np.diff(extended_midpoints)
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Total flux (particles/cm²/s)
        total_flux = np.sum(flux_spectrum * bin_widths)
        metrics['total_flux'] = total_flux
        
        # Calculate dose using flux-to-dose conversion factors
        dose_contribution = np.zeros_like(flux_spectrum)
        for i, energy in enumerate(energy_midpoints):
            dose_contribution[i] = flux_spectrum[i] * bin_widths[i] * flux_to_dose_conversion(energy)
        
        total_dose = np.sum(dose_contribution)
        metrics['total_dose'] = total_dose
        metrics['dose_contribution'] = dose_contribution
        
        # Calculate mean energy (flux-weighted)
        if total_flux > 0:
            mean_energy = np.sum(energy_midpoints * flux_spectrum * bin_widths) / total_flux
        else:
            mean_energy = 0.0
        metrics['mean_energy'] = mean_energy
        
        # Calculate median energy (energy at which cumulative flux = 50% of total)
        if total_flux > 0:
            cumulative_flux = np.cumsum(flux_spectrum * bin_widths)
            normalized_cumulative = cumulative_flux / total_flux
            # Interpolate to find median energy
            try:
                median_energy_interp = interpolate.interp1d(
                    normalized_cumulative, energy_midpoints, bounds_error=False, fill_value="extrapolate")
                median_energy = float(median_energy_interp(0.5))
            except:
                # Fallback if interpolation fails
                idx = np.argmin(np.abs(normalized_cumulative - 0.5))
                median_energy = energy_midpoints[idx]
        else:
            median_energy = 0.0
        metrics['median_energy'] = median_energy
        
        # Energy at peak flux
        peak_idx = np.argmax(flux_spectrum)
        peak_energy = energy_midpoints[peak_idx]
        metrics['peak_energy'] = peak_energy
        metrics['peak_flux'] = flux_spectrum[peak_idx]
        
        # Calculate hardness ratio (high energy flux / low energy flux)
        # Define boundary as 1 MeV
        boundary_idx = np.argmin(np.abs(energy_midpoints - 1.0))
        low_energy_flux = np.sum(flux_spectrum[:boundary_idx] * bin_widths[:boundary_idx])
        high_energy_flux = np.sum(flux_spectrum[boundary_idx:] * bin_widths[boundary_idx:])
        
        if low_energy_flux > 0:
            hardness_ratio = high_energy_flux / low_energy_flux
        else:
            hardness_ratio = float('inf')
        metrics['hardness_ratio'] = hardness_ratio
        
        # Calculate spectrum width (FWHM)
        half_max = flux_spectrum[peak_idx] / 2.0
        try:
            # Find energies where flux is half of maximum
            e_interp = interpolate.interp1d(
                energy_midpoints, flux_spectrum - half_max, bounds_error=False, fill_value="extrapolate")
            
            # Use optimization to find zero crossings
            def find_zero(e_guess):
                return optimize.fsolve(lambda x: e_interp(x), e_guess)[0]
            
            try:
                left_e = find_zero(peak_energy / 2)
                right_e = find_zero(peak_energy * 2)
                fwhm = right_e - left_e
            except:
                # Fallback if optimization fails
                fwhm = 0.0
                
        except:
            # If interpolation fails, estimate using nearest points
            above_half_max = flux_spectrum >= half_max
            if np.sum(above_half_max) > 0:
                fwhm = energy_midpoints[above_half_max][-1] - energy_midpoints[above_half_max][0]
            else:
                fwhm = 0.0
                
        metrics['fwhm'] = fwhm
        
        # Calculate effective dose energy (dose-weighted average energy)
        if total_dose > 0:
            effective_dose_energy = np.sum(energy_midpoints * dose_contribution) / total_dose
        else:
            effective_dose_energy = 0.0
        metrics['effective_dose_energy'] = effective_dose_energy
        
        # Add spectrum data for reference
        metrics['energy_midpoints'] = energy_midpoints
        metrics['flux_spectrum'] = flux_spectrum
        metrics['bin_widths'] = bin_widths
        
        logger.info(f"Spectrum metrics calculated: Total flux={total_flux:.2e}, Mean energy={mean_energy:.2f} MeV")
        
        return metrics

@timeit
def compare_spectra(spectrum1, spectrum2, normalize=True, output_dir=None):
    """
    Compare two energy spectra and calculate similarity metrics.
    
    Parameters:
    -----------
    spectrum1 : dict
        First spectrum data dictionary
    spectrum2 : dict
        Second spectrum data dictionary
    normalize : bool, default=True
        Whether to normalize spectra before comparison
    output_dir : Path or str, optional
        Directory to save comparison plot
        
    Returns:
    --------
    comparison : dict
        Dictionary of comparison metrics
    """
    with LogSection("Comparing spectra"):
        # Extract data
        energy1 = spectrum1['energy_midpoints']
        flux1 = spectrum1['flux_spectrum']
        
        energy2 = spectrum2['energy_midpoints']
        flux2 = spectrum2['flux_spectrum']
        
        # Rebin to common energy grid if necessary
        if not np.array_equal(energy1, energy2):
            logger.info("Rebinning spectra to common energy grid")
            # Create common energy grid (use the finer one)
            if len(energy1) >= len(energy2):
                common_energy = energy1
                # Interpolate spectrum2 to common grid
                flux2_interp = interpolate.interp1d(
                    energy2, flux2, bounds_error=False, fill_value=0.0)(common_energy)
                flux1_common = flux1
            else:
                common_energy = energy2
                # Interpolate spectrum1 to common grid
                flux1_interp = interpolate.interp1d(
                    energy1, flux1, bounds_error=False, fill_value=0.0)(common_energy)
                flux1_common = flux1_interp
                flux2_common = flux2
        else:
            common_energy = energy1
            flux1_common = flux1
            flux2_common = flux2
        
        # Normalize if requested
        if normalize:
            flux1_common = flux1_common / np.sum(flux1_common) if np.sum(flux1_common) > 0 else flux1_common
            flux2_common = flux2_common / np.sum(flux2_common) if np.sum(flux2_common) > 0 else flux2_common
        
        # Calculate comparison metrics
        comparison = {}
        
        # Calculate root mean square error
        rmse = np.sqrt(np.mean((flux1_common - flux2_common)**2))
        comparison['rmse'] = rmse
        
        # Calculate correlation coefficient
        corr = np.corrcoef(flux1_common, flux2_common)[0, 1]
        comparison['correlation'] = corr
        
        # Calculate chi-squared statistic
        nonzero = (flux1_common > 0) & (flux2_common > 0)
        if np.sum(nonzero) > 0:
            chi2 = np.sum(((flux1_common[nonzero] - flux2_common[nonzero])**2) / flux2_common[nonzero])
            comparison['chi_squared'] = chi2
            comparison['chi_squared_reduced'] = chi2 / max(1, np.sum(nonzero) - 1)
        else:
            comparison['chi_squared'] = float('nan')
            comparison['chi_squared_reduced'] = float('nan')
        
        # Calculate area between curves
        area_diff = np.trapz(np.abs(flux1_common - flux2_common), common_energy)
        comparison['area_difference'] = area_diff
        
        # Calculate max absolute difference
        max_diff = np.max(np.abs(flux1_common - flux2_common))
        comparison['max_difference'] = max_diff
        
        # Create comparison plot
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.loglog(common_energy, flux1_common, 'b-', linewidth=2, label='Spectrum 1')
            ax.loglog(common_energy, flux2_common, 'r--', linewidth=2, label='Spectrum 2')
            ax.fill_between(common_energy, flux1_common, flux2_common, alpha=0.3, color='gray')
            
            ax.set_xlabel('Energy (MeV)', fontsize=12)
            ax.set_ylabel('Normalized Flux', fontsize=12) if normalize else ax.set_ylabel('Flux', fontsize=12)
            ax.set_title('Spectrum Comparison', fontsize=14)
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            ax.legend(fontsize=10)
            
            # Add text with comparison metrics
            text = (f"RMSE: {rmse:.2e}\n"
                   f"Correlation: {corr:.2f}\n"
                   f"Chi²/DoF: {comparison['chi_squared_reduced']:.2f}\n"
                                      f"Area Diff: {area_diff:.2e}\n"
                   f"Max Diff: {max_diff:.2e}")
            
            bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8)
            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                  verticalalignment='top', bbox=bbox_props)
            
            plt.tight_layout()
            plt.savefig(output_dir / "spectrum_comparison.png", dpi=300, bbox_inches='tight')
            
            logger.info(f"Spectrum comparison plot saved to {output_dir / 'spectrum_comparison.png'}")
        
        logger.info(f"Spectrum comparison completed: RMSE={rmse:.2e}, Correlation={corr:.2f}")
        
        return comparison

@timeit
def analyze_spectrum_sensitivity(spectra_dict, parameter, output_dir=None):
    """
    Analyze the sensitivity of spectrum to a specific parameter.
    
    Parameters:
    -----------
    spectra_dict : dict
        Dictionary of spectra where keys are parameter values
    parameter : str
        Name of the parameter being analyzed
    output_dir : Path or str, optional
        Directory to save sensitivity plot
        
    Returns:
    --------
    sensitivity : dict
        Dictionary of sensitivity metrics
    """
    with LogSection(f"Analyzing spectrum sensitivity to {parameter}"):
        if len(spectra_dict) < 2:
            logger.warning("Need at least two spectra to analyze sensitivity")
            return {}
        
        # Extract parameter values and sort
        param_values = sorted(list(spectra_dict.keys()))
        base_value = param_values[0]
        base_spectrum = spectra_dict[base_value]
        
        # Calculate metrics for each parameter value
        metrics = {}
        for value in param_values[1:]:
            spectrum = spectra_dict[value]
            comp = compare_spectra(base_spectrum, spectrum, normalize=True)
            metrics[value] = comp
        
        # Calculate overall sensitivity metrics
        sensitivity = {
            'parameter': parameter,
            'base_value': base_value,
            'values': param_values,
            'comparisons': metrics
        }
        
        # Calculate rate of change for each metric
        for metric in ['rmse', 'correlation', 'area_difference', 'max_difference']:
            if all(metric in metrics[v] for v in param_values[1:]):
                values = np.array(param_values[1:])
                metric_values = np.array([metrics[v][metric] for v in values])
                
                # Calculate slope using linear regression
                slope, intercept = np.polyfit(values - base_value, metric_values, 1)
                sensitivity[f'{metric}_slope'] = slope
                
                # Calculate normalized sensitivity (percent change per percent change in parameter)
                if base_value != 0:
                    norm_values = (values - base_value) / base_value
                    if np.mean(metric_values) != 0:
                        norm_sensitivities = (metric_values - metric_values[0]) / np.mean(metric_values)
                        norm_sensitivity = np.mean(norm_sensitivities / norm_values)
                        sensitivity[f'{metric}_normalized_sensitivity'] = norm_sensitivity
        
        # Create sensitivity plot
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Create figure with multiple subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            # Plot metrics vs parameter value
            metrics_to_plot = [
                ('rmse', 'RMSE', 'Root Mean Square Error'),
                ('correlation', 'Correlation', 'Correlation Coefficient'),
                ('area_difference', 'Area Difference', 'Area Between Curves'),
                ('max_difference', 'Max Difference', 'Maximum Absolute Difference')
            ]
            
            for i, (metric, label, title) in enumerate(metrics_to_plot):
                ax = axes[i]
                values = []
                metric_values = []
                
                for value in param_values[1:]:
                    if metric in metrics[value]:
                        values.append(value)
                        metric_values.append(metrics[value][metric])
                
                if values:
                    ax.plot(values, metric_values, 'o-', linewidth=2)
                    ax.set_xlabel(f'{parameter.capitalize()} Value', fontsize=12)
                    ax.set_ylabel(label, fontsize=12)
                    ax.set_title(title, fontsize=14)
                    ax.grid(True, linestyle='--', alpha=0.5)
                    
                    # Add trendline
                    try:
                        z = np.polyfit(values, metric_values, 1)
                        p = np.poly1d(z)
                        ax.plot(values, p(values), "r--", alpha=0.7)
                        ax.text(0.05, 0.95, f"Slope: {z[0]:.2e}", transform=ax.transAxes,
                               fontsize=10, verticalalignment='top')
                    except:
                        pass
            
            plt.tight_layout()
            plt.savefig(output_dir / f"{parameter}_sensitivity.png", dpi=300, bbox_inches='tight')
            
            # Also create a spectrum overlay plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for value in param_values:
                spectrum = spectra_dict[value]
                energy = spectrum['energy_midpoints']
                flux = spectrum['flux_spectrum']
                
                # Normalize flux
                norm_flux = flux / np.sum(flux) if np.sum(flux) > 0 else flux
                
                ax.loglog(energy, norm_flux, linewidth=2, label=f"{parameter}={value}")
            
            ax.set_xlabel('Energy (MeV)', fontsize=12)
            ax.set_ylabel('Normalized Flux', fontsize=12)
            ax.set_title(f'Spectrum Sensitivity to {parameter.capitalize()}', fontsize=14)
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            ax.legend(fontsize=10)
            
            plt.tight_layout()
            plt.savefig(output_dir / f"{parameter}_spectra_overlay.png", dpi=300, bbox_inches='tight')
            
            logger.info(f"Sensitivity plots saved to {output_dir}")
        
        logger.info(f"Spectrum sensitivity analysis completed for {parameter}")
        
        return sensitivity

@timeit
def extract_characteristic_energies(spectrum_data):
    """
    Extract characteristic energies from a spectrum.
    
    Parameters:
    -----------
    spectrum_data : dict
        Dictionary containing spectrum data
        
    Returns:
    --------
    energies : dict
        Dictionary of characteristic energies
    """
    with LogSection("Extracting characteristic energies"):
        # Extract data
        energy_midpoints = spectrum_data['energy_midpoints']
        flux_spectrum = spectrum_data['flux_spectrum']
        
        # Initialize result dictionary
        energies = {}
        
        # Calculate total flux and cumulative distribution
        if 'bin_widths' in spectrum_data:
            bin_widths = spectrum_data['bin_widths']
        else:
            # Estimate bin widths from midpoints
            extended_midpoints = np.concatenate([
                [energy_midpoints[0] - (energy_midpoints[1] - energy_midpoints[0])/2],
                energy_midpoints,
                [energy_midpoints[-1] + (energy_midpoints[-1] - energy_midpoints[-2])/2]
            ])
            bin_widths = np.diff(extended_midpoints)
        
        total_flux = np.sum(flux_spectrum * bin_widths)
        cum_flux = np.cumsum(flux_spectrum * bin_widths)
        norm_cum_flux = cum_flux / total_flux if total_flux > 0 else cum_flux
        
        # Peak energy (modal energy)
        peak_idx = np.argmax(flux_spectrum)
        energies['peak'] = energy_midpoints[peak_idx]
        
        # Extract quantile energies (E10, E50, E90)
        quantiles = [0.1, 0.5, 0.9]
        for q in quantiles:
            try:
                # Interpolate to find quantile energy
                quantile_interp = interpolate.interp1d(
                    norm_cum_flux, energy_midpoints, bounds_error=False, fill_value="extrapolate")
                energies[f'E{int(q*100)}'] = float(quantile_interp(q))
            except:
                # Fallback if interpolation fails
                idx = np.argmin(np.abs(norm_cum_flux - q))
                energies[f'E{int(q*100)}'] = energy_midpoints[idx]
        
        # Mean energy (flux-weighted average)
        if total_flux > 0:
            mean_energy = np.sum(energy_midpoints * flux_spectrum * bin_widths) / total_flux
        else:
            mean_energy = 0.0
        energies['mean'] = mean_energy
        
        # Find cutoff energy (energy above which flux drops to 1% of peak)
        threshold = 0.01 * flux_spectrum[peak_idx]
        above_threshold = flux_spectrum >= threshold
        if np.any(above_threshold & (energy_midpoints > energies['peak'])):
            cutoff_idx = np.max(np.where(above_threshold & (energy_midpoints > energies['peak']))[0])
            energies['cutoff'] = energy_midpoints[cutoff_idx]
        else:
            energies['cutoff'] = energies['peak']
        
        # Energy width (difference between 10% and 90% quantiles)
        energies['width'] = energies['E90'] - energies['E10']
        
        logger.info(f"Characteristic energies extracted: Peak={energies['peak']:.2f} MeV, "
                  f"Median={energies['E50']:.2f} MeV, Mean={energies['mean']:.2f} MeV")
        
        return energies

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Analyze energy spectra from simulation results")
    parser.add_argument("--results", type=str, required=True, help="Path to results JSON file")
    parser.add_argument("--output", type=str, default=None, help="Output directory for plots and data")
    
    args = parser.parse_args()
    
    # Load results
    with open(args.results, 'r') as f:
        results = json.load(f)
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = PLOTS_DIR / "spectrum_analysis"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Process each result
    all_metrics = []
    for i, result in enumerate(results):
        if 'energy_midpoints' in result and 'flux_spectrum' in result:
            params = result['parameters']
            
            logger.info(f"Analyzing spectrum {i+1}/{len(results)}: "
                      f"E={params['energy']} MeV, D={params['channel_diameter']} cm, "
                      f"Distance={params['detector_distance']} cm, Angle={params['detector_angle']}°")
            
            # Calculate spectrum metrics
            metrics = calculate_spectrum_metrics(result)
            
            # Extract characteristic energies
            energies = extract_characteristic_energies(result)
            
            # Save metrics and energies to result
            result['spectrum_metrics'] = metrics
            result['characteristic_energies'] = energies
            
            # Record metrics with parameters for summary
            metrics_with_params = {
                'energy': params['energy'],
                'channel_diameter': params['channel_diameter'],
                'detector_distance': params['detector_distance'],
                'detector_angle': params['detector_angle'],
                **{f'spectrum_{k}': v for k, v in metrics.items() if not isinstance(v, (list, np.ndarray))},
                **{f'energy_{k}': v for k, v in energies.items()}
            }
            all_metrics.append(metrics_with_params)
    
    # Create summary dataframe
    df = pd.DataFrame(all_metrics)
    df.to_csv(output_dir / "spectrum_metrics_summary.csv", index=False)
    
    logger.info(f"Spectrum analysis complete. Results saved to {output_dir}")
