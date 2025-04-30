#!/usr/bin/env python3
"""
Error analysis and uncertainty quantification for gamma ray streaming simulations.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import time
from logging_utils import logger, LogSection, timeit

def calculate_statistical_error(tally_result):
    """
    Calculate statistical error from OpenMC tally results.
    
    Parameters:
    -----------
    tally_result : openmc.Tally
        OpenMC tally result
    
    Returns:
    --------
    rel_error : float
        Relative error (statistical uncertainty)
    """
    # Extract mean and standard deviation
    mean = tally_result.mean
    std_dev = tally_result.std_dev
    
    # Calculate relative error
    rel_error = std_dev / mean if mean > 0 else np.nan
    
    return rel_error

def propagate_errors(values, errors, operation='sum'):
    """
    Propagate errors through mathematical operations.
    
    Parameters:
    -----------
    values : array-like
        Array of values
    errors : array-like
        Array of absolute errors corresponding to values
    operation : str, default='sum'
                Mathematical operation to perform ('sum', 'product', 'mean', 'ratio')
    
    Returns:
    --------
    result : float
        Result of the operation
    result_error : float
        Propagated error in the result
    """
    # Convert inputs to numpy arrays
    values = np.asarray(values)
    errors = np.asarray(errors)
    
    if operation == 'sum':
        # Sum of values
        result = np.sum(values)
        # Error propagation for sum: sqrt(sum of squared errors)
        result_error = np.sqrt(np.sum(errors**2))
        
    elif operation == 'product':
        # Product of values
        result = np.prod(values)
        # Error propagation for product: product * sqrt(sum of squared relative errors)
        rel_errors = errors / values
        result_error = result * np.sqrt(np.sum(rel_errors**2))
        
    elif operation == 'mean':
        # Mean of values
        result = np.mean(values)
        # Error propagation for mean: sqrt(sum of squared errors) / n
        result_error = np.sqrt(np.sum(errors**2)) / len(values)
        
    elif operation == 'ratio':
        # Ensure we have exactly two values for ratio
        if len(values) != 2:
            raise ValueError("Ratio operation requires exactly two values")
        
        # Ratio of values
        result = values[0] / values[1]
        # Error propagation for ratio: |result| * sqrt((errors[0]/values[0])^2 + (errors[1]/values[1])^2)
        result_error = abs(result) * np.sqrt((errors[0]/values[0])**2 + (errors[1]/values[1])**2)
        
    else:
        raise ValueError(f"Unsupported operation: {operation}")
    
    return result, result_error

@timeit
def analyze_convergence(results, parameter_key=None):
    """
    Analyze convergence of simulation results with respect to a parameter.
    
    Parameters:
    -----------
    results : list
        List of simulation results
    parameter_key : str, optional
        Key for the parameter to analyze convergence for
        
    Returns:
    --------
    convergence : dict
        Dictionary with convergence metrics
    """
    with LogSection("Analyzing convergence"):
        if not results:
            logger.warning("No results provided for convergence analysis")
            return {}
        
        # If no parameter specified, try to find one with multiple values
        if parameter_key is None:
            # Check parameters that might have multiple values
            potential_params = ['n_particles', 'n_batches', 'energy_groups']
            
            for param in potential_params:
                # Extract all values of this parameter
                param_values = [r['parameters'].get(param, None) for r in results if 'parameters' in r]
                # Filter out None values
                param_values = [v for v in param_values if v is not None]
                
                if len(set(param_values)) > 1:
                    parameter_key = param
                    logger.info(f"Using {parameter_key} for convergence analysis")
                    break
        
        if parameter_key is None:
            logger.warning("Could not find a suitable parameter for convergence analysis")
            return {}
        
        # Group results by parameter value
        grouped_results = {}
        for result in results:
            if 'parameters' in result and parameter_key in result['parameters']:
                param_value = result['parameters'][parameter_key]
                
                if param_value not in grouped_results:
                    grouped_results[param_value] = []
                
                grouped_results[param_value].append(result)
        
        # Sort parameter values
        param_values = sorted(grouped_results.keys())
        
        if len(param_values) < 2:
            logger.warning(f"Need at least two different values of {parameter_key} for convergence analysis")
            return {}
        
        # Initialize convergence metrics
        convergence = {
            'parameter': parameter_key,
            'values': param_values,
            'total_dose': [],
            'total_dose_error': [],
            'convergence_ratio': [],
            'relative_difference': []
        }
        
        # Extract average results for each parameter value
        for param_value in param_values:
            param_results = grouped_results[param_value]
            
            # Calculate average total dose for this parameter value
            total_doses = [r.get('total_dose', 0) for r in param_results]
            total_dose_errors = [r.get('total_dose_error', 0) for r in param_results]
            
            avg_dose = np.mean(total_doses)
            avg_error = np.sqrt(np.sum(np.array(total_dose_errors)**2)) / len(total_dose_errors)
            
            convergence['total_dose'].append(avg_dose)
            convergence['total_dose_error'].append(avg_error)
        
        # Calculate convergence metrics
        for i in range(1, len(param_values)):
            # Convergence ratio: error reduction with increasing parameter value
            error_ratio = convergence['total_dose_error'][i-1] / convergence['total_dose_error'][i] if convergence['total_dose_error'][i] > 0 else np.nan
            convergence['convergence_ratio'].append(error_ratio)
            
            # Relative difference between consecutive values
            rel_diff = abs(convergence['total_dose'][i] - convergence['total_dose'][i-1]) / convergence['total_dose'][i-1] if convergence['total_dose'][i-1] > 0 else np.nan
            convergence['relative_difference'].append(rel_diff)
        
        # Calculate convergence order
        try:
            if len(param_values) >= 3:
                # Use last three points for convergence order estimation
                x = np.log(param_values[-3:])
                y = np.log(convergence['total_dose_error'][-3:])
                
                # Linear regression
                slope, _, r_value, _, _ = stats.linregress(x, y)
                
                # Convergence order is negative of slope
                convergence_order = -slope
                convergence['convergence_order'] = convergence_order
                convergence['convergence_r_squared'] = r_value**2
                
                logger.info(f"Estimated convergence order: {convergence_order:.2f} (R²={r_value**2:.3f})")
        except:
            logger.warning("Could not estimate convergence order")
        
        return convergence

@timeit
def estimate_minimum_particles(results, target_error=0.05):
    """
    Estimate minimum number of particles needed to achieve a target error level.
    
    Parameters:
    -----------
    results : list
        List of simulation results
    target_error : float, default=0.05
        Target relative error (e.g., 0.05 for 5%)
        
    Returns:
    --------
    min_particles : dict
        Dictionary with estimation results
    """
    with LogSection(f"Estimating minimum particles for {target_error*100:.1f}% error"):
        # Extract results with 'n_particles' and 'total_dose_error' information
        valid_results = []
        for result in results:
            if ('parameters' in result and 'n_particles' in result['parameters'] and 
                'total_dose_error' in result and 'total_dose' in result):
                
                n_particles = result['parameters']['n_particles']
                rel_error = result['total_dose_error'] / result['total_dose'] if result['total_dose'] > 0 else np.nan
                
                if not np.isnan(rel_error):
                    valid_results.append({
                        'n_particles': n_particles,
                        'rel_error': rel_error
                    })
        
        if len(valid_results) < 2:
            logger.warning("Insufficient data to estimate minimum particles")
            return {}
        
        # Sort by number of particles
        valid_results.sort(key=lambda x: x['n_particles'])
        
        # Extract arrays for fit
        n_particles = np.array([r['n_particles'] for r in valid_results])
        rel_errors = np.array([r['rel_error'] for r in valid_results])
        
        # For Monte Carlo simulations, error ~ 1/sqrt(N)
        # So we expect rel_error = C / sqrt(n_particles)
        # log(rel_error) = log(C) - 0.5 * log(n_particles)
        
        log_n = np.log(n_particles)
        log_err = np.log(rel_errors)
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_err)
        
        # Expected slope should be -0.5 for Monte Carlo convergence
        expected_slope = -0.5
        
        # Calculate constants for estimated relation
        C = np.exp(intercept)
        
        # Estimate minimum particles for target error
        # rel_error = C / sqrt(n_particles)
        # n_particles = (C / rel_error)^2
        estimated_n_particles = (C / target_error)**2
        
        # Round up to nearest 10,000
        min_particles = int(np.ceil(estimated_n_particles / 10000) * 10000)
        
        # Create result dictionary
        result = {
            'min_particles': min_particles,
            'target_error': target_error,
            'coefficient': C,
            'fit_slope': slope,
            'fit_intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err,
            'expected_slope': expected_slope,
            'slope_difference': slope - expected_slope
        }
        
        logger.info(f"Estimated {min_particles:,} particles needed for {target_error*100:.1f}% error")
        
        if abs(slope - expected_slope) > 0.1:
            logger.warning(f"Fitted slope ({slope:.3f}) deviates from expected Monte Carlo behavior (-0.5)")
        
        return result

@timeit
def plot_error_analysis(results, output_dir=None):
    """
    Create plots for error analysis.
    
    Parameters:
    -----------
    results : list
        List of simulation results
    output_dir : str or Path, optional
        Directory to save plots
        
    Returns:
    --------
    figs : dict
        Dictionary of matplotlib figures
    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    
    with LogSection("Creating error analysis plots"):
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        
        figs = {}
        
        # Extract relevant data for error analysis
        valid_results = []
        for result in results:
            if ('parameters' in result and 'n_particles' in result['parameters'] and 
                'total_dose_error' in result and 'total_dose' in result):
                
                n_particles = result['parameters']['n_particles']
                rel_error = result['total_dose_error'] / result['total_dose'] if result['total_dose'] > 0 else np.nan
                
                if not np.isnan(rel_error):
                    valid_results.append({
                        'n_particles': n_particles,
                        'rel_error': rel_error,
                        'energy': result['parameters'].get('energy', 0),
                        'channel_diameter': result['parameters'].get('channel_diameter', 0),
                        'detector_distance': result['parameters'].get('detector_distance', 0)
                    })
        
        if not valid_results:
            logger.warning("No valid results for error analysis plots")
            return figs
        
        # 1. Relative error vs. number of particles
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        df = pd.DataFrame(valid_results)
        particle_groups = df.groupby('n_particles')
        
        n_particles_array = np.array(sorted(df['n_particles'].unique()))
        mean_errors = np.array([particle_groups.get_group(n)['rel_error'].mean() for n in n_particles_array])
        std_errors = np.array([particle_groups.get_group(n)['rel_error'].std() for n in n_particles_array])
        
        ax1.errorbar(n_particles_array, mean_errors, yerr=std_errors, fmt='o-', linewidth=2, 
                    elinewidth=1, capsize=4, label='Simulation Results')
        
        # Add theoretical 1/sqrt(N) line
        if len(mean_errors) > 0:
            scale_factor = mean_errors[0] * np.sqrt(n_particles_array[0])
            theoretical_errors = scale_factor / np.sqrt(n_particles_array)
            ax1.plot(n_particles_array, theoretical_errors, 'r--', linewidth=2, 
                    label='Theoretical (1/√N scaling)')
        
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Number of Particles', fontsize=12)
        ax1.set_ylabel('Relative Error', fontsize=12)
        ax1.set_title('Convergence of Relative Error with Particle Count', fontsize=14)
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        ax1.legend(fontsize=10)
        
        if output_dir is not None:
            plt.savefig(output_dir / "error_vs_particles.png", dpi=300, bbox_inches='tight')
        
        figs['error_vs_particles'] = fig1
        
        # 2. Error distribution analysis
        if len(valid_results) >= 5:  # Need enough data for histogram
            rel_errors = np.array([r['rel_error'] for r in valid_results])
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            # Histogram of relative errors
            ax2.hist(rel_errors, bins=20, alpha=0.7, color='b', edgecolor='black')
            
            # Add normal distribution fit
            mu, sigma = np.mean(rel_errors), np.std(rel_errors)
            x = np.linspace(max(0, mu - 3*sigma), mu + 3*sigma, 100)
            y = stats.norm.pdf(x, mu, sigma) * len(rel_errors) * (x[1] - x[0])  # Scale PDF to match histogram
            ax2.plot(x, y, 'r-', linewidth=2, label=f'Normal: μ={mu:.4f}, σ={sigma:.4f}')
            
            ax2.set_xlabel('Relative Error', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Distribution of Relative Errors', fontsize=14)
            ax2.grid(True, linestyle='--', alpha=0.5)
            ax2.legend(fontsize=10)

                        
            if output_dir is not None:
                plt.savefig(output_dir / "error_distribution.png", dpi=300, bbox_inches='tight')
            
            figs['error_distribution'] = fig2
        
        # 3. Error vs. parameter analysis - create scatter plots for each relevant parameter
        parameters = ['energy', 'channel_diameter', 'detector_distance']
        for param in parameters:
            # Skip if parameter doesn't have multiple values
            if len(set(r[param] for r in valid_results if param in r)) <= 1:
                continue
            
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            
            # Group by parameter value
            param_df = pd.DataFrame(valid_results)
            scatter = ax3.scatter(param_df[param], param_df['rel_error'], 
                                c=np.log10(param_df['n_particles']), cmap='viridis', 
                                alpha=0.7, s=50, edgecolor='k')
            
            # Add colorbar for particle count
            cbar = plt.colorbar(scatter)
            cbar.set_label('log10(Particle Count)', fontsize=10)
            
            # Add trendline
            try:
                z = np.polyfit(param_df[param], param_df['rel_error'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(param_df[param]), max(param_df[param]), 100)
                ax3.plot(x_trend, p(x_trend), "r--", alpha=0.7)
            except:
                pass
            
            ax3.set_xlabel(f'{param.replace("_", " ").title()}', fontsize=12)
            ax3.set_ylabel('Relative Error', fontsize=12)
            ax3.set_title(f'Error vs. {param.replace("_", " ").title()}', fontsize=14)
            ax3.grid(True, linestyle='--', alpha=0.5)
            
            if output_dir is not None:
                plt.savefig(output_dir / f"error_vs_{param}.png", dpi=300, bbox_inches='tight')
            
            figs[f'error_vs_{param}'] = fig3
        
        logger.info(f"Created {len(figs)} error analysis plots")
        
        return figs

@timeit
def analyze_error_sources(results):
    """
    Analyze different sources of error in simulation results.
    
    Parameters:
    -----------
    results : list
        List of simulation results
        
    Returns:
    --------
    error_sources : dict
        Dictionary of error source analysis
    """
    with LogSection("Analyzing error sources"):
        if not results:
            logger.warning("No results provided for error source analysis")
            return {}
        
        # Initialize error sources dictionary
        error_sources = {
            'statistical': [],
            'modeling': [],
            'discretization': [],
            'cross_section': [],
            'total': []
        }
        
        # Statistical error - from Monte Carlo tallies
        for result in results:
            if 'total_dose' in result and 'total_dose_error' in result:
                rel_statistical_error = result['total_dose_error'] / result['total_dose'] if result['total_dose'] > 0 else np.nan
                
                if not np.isnan(rel_statistical_error):
                    error_sources['statistical'].append({
                        'value': rel_statistical_error,
                        'parameters': result.get('parameters', {})
                    })
        
        # Modeling error - estimated by comparing different modeling approaches
        # For example, comparing results with different concrete compositions
        concrete_types = set()
        for result in results:
            if 'parameters' in result and 'concrete_type' in result['parameters']:
                concrete_types.add(result['parameters']['concrete_type'])
        
        if len(concrete_types) > 1:
            # Group results by parameter sets (excluding concrete_type)
            grouped_results = {}
            for result in results:
                if 'parameters' not in result or 'total_dose' not in result:
                    continue
                
                # Create key tuple from parameters (excluding concrete_type)
                param_dict = result['parameters'].copy()
                concrete_type = param_dict.pop('concrete_type', None)
                
                # Convert dict to tuple of items for hashing
                param_key = tuple(sorted(param_dict.items()))
                
                if param_key not in grouped_results:
                    grouped_results[param_key] = {}
                
                grouped_results[param_key][concrete_type] = result['total_dose']
            
            # Calculate modeling error for each parameter set
            for param_key, type_results in grouped_results.items():
                if len(type_results) > 1:
                    dose_values = list(type_results.values())
                    mean_dose = np.mean(dose_values)
                    max_deviation = np.max(np.abs(np.array(dose_values) - mean_dose))
                    rel_modeling_error = max_deviation / mean_dose if mean_dose > 0 else np.nan
                    
                    if not np.isnan(rel_modeling_error):
                        # Convert param_key back to dict
                        params = dict(param_key)
                        error_sources['modeling'].append({
                            'value': rel_modeling_error,
                            'parameters': params,
                            'concrete_types': list(type_results.keys())
                        })
        
        # Discretization error - estimated by comparing different mesh resolutions
        mesh_resolutions = set()
        for result in results:
            if 'parameters' in result and 'mesh_resolution' in result['parameters']:
                mesh_resolutions.add(result['parameters']['mesh_resolution'])
        
        if len(mesh_resolutions) > 1:
            # Similar approach as for modeling error
            grouped_results = {}
            for result in results:
                if 'parameters' not in result or 'total_dose' not in result:
                    continue
                
                # Create key tuple from parameters (excluding mesh_resolution)
                param_dict = result['parameters'].copy()
                mesh_resolution = param_dict.pop('mesh_resolution', None)
                
                param_key = tuple(sorted(param_dict.items()))
                
                if param_key not in grouped_results:
                    grouped_results[param_key] = {}
                
                grouped_results[param_key][mesh_resolution] = result['total_dose']
            
            # Calculate discretization error
            for param_key, mesh_results in grouped_results.items():
                if len(mesh_results) > 1:
                    # Sort mesh resolutions
                    sorted_meshes = sorted(mesh_results.keys())
                    finest_mesh = sorted_meshes[-1]
                    
                    # Use finest mesh as reference
                    reference_dose = mesh_results[finest_mesh]
                    
                    for mesh in sorted_meshes[:-1]:
                        dose = mesh_results[mesh]
                        rel_disc_error = abs(dose - reference_dose) / reference_dose if reference_dose > 0 else np.nan
                        
                        if not np.isnan(rel_disc_error):
                            params = dict(param_key)
                            error_sources['discretization'].append({
                                'value': rel_disc_error,
                                'parameters': params,
                                'mesh_resolution': mesh,
                                'reference_mesh': finest_mesh
                            })
        
        # Estimate cross-section error (if multiple libraries used)
        xs_libraries = set()
        for result in results:
            if 'parameters' in result and 'cross_section_library' in result['parameters']:
                xs_libraries.add(result['parameters']['cross_section_library'])
        
        if len(xs_libraries) > 1:
            # Similar approach as above
            grouped_results = {}
            for result in results:
                if 'parameters' not in result or 'total_dose' not in result:
                    continue
                
                param_dict = result['parameters'].copy()
                xs_library = param_dict.pop('cross_section_library', None)
                
                param_key = tuple(sorted(param_dict.items()))
                
                if param_key not in grouped_results:
                    grouped_results[param_key] = {}
                
                grouped_results[param_key][xs_library] = result['total_dose']
            
            # Calculate cross-section error
            for param_key, xs_results in grouped_results.items():
                if len(xs_results) > 1:
                    dose_values = list(xs_results.values())
                    mean_dose = np.mean(dose_values)
                    max_deviation = np.max(np.abs(np.array(dose_values) - mean_dose))
                    rel_xs_error = max_deviation / mean_dose if mean_dose > 0 else np.nan
                    
                    if not np.isnan(rel_xs_error):
                        params = dict(param_key)
                        error_sources['cross_section'].append({
                            'value': rel_xs_error,
                            'parameters': params,
                            'xs_libraries': list(xs_results.keys())
                        })
        
        # Calculate total error for each result (combining available error sources)
        for result in results:
            if 'parameters' not in result or 'total_dose' not in result:
                continue
            
            # Statistical error is always available
            if 'total_dose_error' in result:
                rel_stat_error = (result['total_dose_error'] / result['total_dose']) if result['total_dose'] > 0 else 0.0
                rel_stat_error = rel_stat_error**2  # Square for quadratic combination
            else:
                rel_stat_error = 0.0
            
            # Find matching modeling error if available
            rel_model_error = 0.0
            for entry in error_sources['modeling']:
                if all(result['parameters'].get(k) == v for k, v in entry['parameters'].items()):
                    rel_model_error = entry['value']**2
                    break
            
            # Find matching discretization error if available
            rel_disc_error = 0.0
            for entry in error_sources['discretization']:
                if all(result['parameters'].get(k) == v for k, v in entry['parameters'].items()):
                    rel_disc_error = entry['value']**2
                    break
            
            # Find matching cross-section error if available
            rel_xs_error = 0.0
            for entry in error_sources['cross_section']:
                if all(result['parameters'].get(k) == v for k, v in entry['parameters'].items()):
                    rel_xs_error = entry['value']**2
                    break
            
            # Combine errors in quadrature
            total_rel_error = np.sqrt(rel_stat_error + rel_model_error + rel_disc_error + rel_xs_error)
            
            error_sources['total'].append({
                'value': total_rel_error,
                'parameters': result['parameters'],
                'components': {
                    'statistical': np.sqrt(rel_stat_error),
                    'modeling': np.sqrt(rel_model_error),
                    'discretization': np.sqrt(rel_disc_error),
                    'cross_section': np.sqrt(rel_xs_error)
                }
            })
        
        # Calculate summary statistics for each error source
        for source in error_sources:
            if error_sources[source]:
                values = [entry['value'] for entry in error_sources[source]]
                error_sources[f'{source}_mean'] = np.mean(values)
                error_sources[f'{source}_median'] = np.median(values)
                error_sources[f'{source}_min'] = np.min(values)
                error_sources[f'{source}_max'] = np.max(values)
                error_sources[f'{source}_std'] = np.std(values)
        
        return error_sources

if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Perform error analysis on simulation results")
    parser.add_argument("--results", type=str, required=True, help="Path to results JSON file")
    parser.add_argument("--output", type=str, default=None, help="Output directory for plots and data")
    parser.add_argument("--target-error", type=float, default=0.05, help="Target relative error for particle estimation")
    
    args = parser.parse_args()
    
    # Load results
    with open(args.results, 'r') as f:
        results = json.load(f)
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        from config import PLOTS_DIR
        output_dir = PLOTS_DIR / "error_analysis"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Perform analyses
    convergence = analyze_convergence(results)
    min_particles = estimate_minimum_particles(results, target_error=args.target_error)
    plot_error_analysis(results, output_dir=output_dir)
    error_sources = analyze_error_sources(results)
    
    # Save results
    analysis_results = {
        'convergence': convergence,
        'min_particles': min_particles,
        'error_sources': error_sources
    }
    
    with open(output_dir / "error_analysis.json", 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Generate summary report
    with open(output_dir / "error_analysis_summary.md", 'w') as f:
        f.write("# Error Analysis Summary\n\n")
        
        f.write("## Convergence Analysis\n\n")
        if 'convergence_order' in convergence:
            f.write(f"- Estimated convergence order: {convergence['convergence_order']:.3f}\n")
            f.write(f"- R-squared: {convergence['convergence_r_squared']:.3f}\n")
        
        f.write("\n## Minimum Particle Estimation\n\n")
        f.write(f"- Target error: {args.target_error*100:.1f}%\n")
        if 'min_particles' in min_particles:
            f.write(f"- Estimated minimum particles: {min_particles['min_particles']:,}\n")
            f.write(f"- Fitted error relation: error = {min_particles['coefficient']:.3e} / sqrt(N)\n")
        
        f.write("\n## Error Sources\n\n")
        sources = ['statistical', 'modeling', 'discretization', 'cross_section', 'total']
        for source in sources:
            if f'{source}_mean' in error_sources:
                f.write(f"### {source.capitalize()} Error\n\n")
                f.write(f"- Mean: {error_sources[f'{source}_mean']*100:.2f}%\n")
                f.write(f"- Median: {error_sources[f'{source}_median']*100:.2f}%\n")
                f.write(f"- Range: {error_sources[f'{source}_min']*100:.2f}% - {error_sources[f'{source}_max']*100:.2f}%\n\n")
    
        logger.info(f"Error analysis complete. Results saved to {output_dir}")



