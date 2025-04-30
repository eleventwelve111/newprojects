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
        Array of errors corresponding to values
    operation : str
        Mathematical operation ('sum', 'product', 'ratio', 'exp')
    
    Returns:
    --------
    result_value : float
        Result of operation on values
    result_error : float
        Propagated error
    """
    values = np.array(values)
    errors = np.array(errors)
    
    if operation == 'sum':
        # Error propagation for sum: σ_sum = sqrt(sum(σ_i^2))
        result_value = np.sum(values)
        result_error = np.sqrt(np.sum(errors**2))
        
    elif operation == 'product':
        # Error propagation for product: (σ_prod/prod)^2 = sum((σ_i/x_i)^2)
        result_value = np.prod(values)
        rel_errors_squared = np.sum((errors/values)**2) if all(values != 0) else np.nan
        result_error = result_value * np.sqrt(rel_errors_squared) if not np.isnan(rel_errors_squared) else np.nan
        
    elif operation == 'ratio':
        # Requires exactly two values: value1/value2
        if len(values) != 2:
            raise ValueError("Ratio operation requires exactly two values")
        
        # Error propagation for ratio: σ_(a/b)/(a/b) = sqrt((σ_a/a)^2 + (σ_b/b)^2)
        result_value = values[0] / values[1] if values[1] != 0 else np.nan
        
        if result_value is not np.nan:
            rel_error = np.sqrt((errors[0]/values[0])**2 + (errors[1]/values[1])**2) if values[0] != 0 and values[1] != 0 else np.nan
            result_error = result_value * rel_error if not np.isnan(rel_error) else np.nan
        else:
            result_error = np.nan
            
    elif operation == 'exp':
        # Error propagation for exponential: σ_exp(x) = exp(x) * σ_x
        result_value = np.exp(values[0])
        result_error = result_value * errors[0]
        
    else:
        raise ValueError(f"Unsupported operation: {operation}")
        
    return result_value, result_error

@timeit
def analyze_simulation_errors(results):
    """
    Analyze errors in simulation results.
    
    Parameters:
    -----------
    results : dict
        Dictionary of simulation results with errors
        
    Returns:
    --------
    error_analysis : dict
        Dictionary with error analysis results
    """
    with LogSection("Analyzing simulation errors"):
        error_analysis = {}
        
        # Extract all errors for each tally type
        flux_errors = []
        dose_errors = []
        kerma_errors = []
        heating_errors = []
        
        for scenario, data in results.items():
            if 'flux' in data and 'flux_error' in data:
                flux_errors.append(data['flux_error'])
            if 'dose' in data and 'dose_error' in data:
                dose_errors.append(data['dose_error'])
            if 'kerma' in data and 'kerma_error' in data:
                kerma_errors.append(data['kerma_error'])
            if 'heating' in data and 'heating_error' in data:
                heating_errors.append(data['heating_error'])
        
        # Calculate error statistics
        error_analysis['flux'] = {
            'mean_error': np.mean(flux_errors) if flux_errors else np.nan,
            'max_error': np.max(flux_errors) if flux_errors else np.nan,
            'min_error': np.min(flux_errors) if flux_errors else np.nan,
            'std_dev': np.std(flux_errors) if flux_errors else np.nan
        }
        
        error_analysis['dose'] = {
            'mean_error': np.mean(dose_errors) if dose_errors else np.nan,
            'max_error': np.max(dose_errors) if dose_errors else np.nan,
            'min_error': np.min(dose_errors) if dose_errors else np.nan,
            'std_dev': np.std(dose_errors) if dose_errors else np.nan
        }
        
        error_analysis['kerma'] = {
            'mean_error': np.mean(kerma_errors) if kerma_errors else np.nan,
            'max_error': np.max(kerma_errors) if kerma_errors else np.nan,
            'min_error': np.min(kerma_errors) if kerma_errors else np.nan,
            'std_dev': np.std(kerma_errors) if kerma_errors else np.nan
        }
        
        error_analysis['heating'] = {
            'mean_error': np.mean(heating_errors) if heating_errors else np.nan,
            'max_error': np.max(heating_errors) if heating_errors else np.nan,
            'min_error': np.min(heating_errors) if heating_errors else np.nan,
            'std_dev': np.std(heating_errors) if heating_errors else np.nan
        }
        
        # Log summary of errors
        logger.info("Error analysis summary:")
        for tally_type, stats in error_analysis.items():
            logger.info(f"{tally_type.capitalize()} errors: Mean = {stats['mean_error']:.4f}, Max = {stats['max_error']:.4f}")
        
        return error_analysis

def plot_error_distribution(results, tally_type='dose'):
    """
    Plot the distribution of errors for a specific tally type.
    
    Parameters:
    -----------
    results : dict
        Dictionary of simulation results with errors
    tally_type : str
        Type of tally to analyze ('flux', 'dose', 'kerma', 'heating')
    
    Returns:
    --------
    fig : matplotlib.Figure
        Figure containing the error distribution plot
    """
    error_key = f'{tally_type}_error'
    errors = []
    
    for scenario, data in results.items():
        if error_key in data:
            errors.append(data[error_key])
    
    if not errors:
        logger.warning(f"No error data found for '{tally_type}'")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    counts, bins, _ = ax.hist(errors, bins=20, alpha=0.6, color='blue', edgecolor='black')
    
    # Add kernel density estimate
    if len(errors) > 5:  # Need enough data points for KDE
        x = np.linspace(min(errors), max(errors), 1000)
        kde = stats.gaussian_kde(errors)
        ax.plot(x, kde(x) * (bins[1] - bins[0]) * len(errors), 'r-', lw=2, 
                label='Kernel Density Estimate')
    
    # Add vertical line for mean and median
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    ax.axvline(mean_error, color='green', linestyle='--', lw=2, label=f'Mean: {mean_error:.4f}')
    ax.axvline(median_error, color='orange', linestyle='--', lw=2, label=f'Median: {median_error:.4f}')
    
    # Add labels and title
    ax.set_xlabel('Relative Error')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {tally_type.capitalize()} Errors')
    ax.legend()
    
    return fig

def analyze_convergence(batch_results):
    """
    Analyze the convergence of results over batches.
    
    Parameters:
    -----------
    batch_results : dict
        Dictionary with results for each batch
    
    Returns:
    --------
    convergence_metrics : dict
        Dictionary with convergence metrics
    """
    convergence_metrics = {}
    
    # Extract batch numbers and results
    batches = sorted(batch_results.keys())
    
    if not batches:
        logger.warning("No batch data available for convergence analysis")
        return convergence_metrics
    
    # Extract values for different tally types
    tally_types = ['flux', 'dose', 'kerma', 'heating']
    
    for tally_type in tally_types:
        values = []
        errors = []
        
        for batch in batches:
            if tally_type in batch_results[batch] and f'{tally_type}_error' in batch_results[batch]:
                values.append(batch_results[batch][tally_type])
                errors.append(batch_results[batch][f'{tally_type}_error'])
        
        if values:
            # Calculate convergence metrics
            final_value = values[-1]
            converged_at_batch = None
            
            # Consider convergence achieved when values are within 5% of final value
            for i, value in enumerate(values):
                if abs((value - final_value) / final_value) < 0.05:
                    converged_at_batch = batches[i]
                    break
            
            # Record convergence metrics
            convergence_metrics[tally_type] = {
                'final_value': final_value,
                'converged_at_batch': converged_at_batch,
                'values': values,
                'errors': errors,
                'batches': batches
            }
    
    # Log convergence summary
    for tally_type, metrics in convergence_metrics.items():
        if 'converged_at_batch' in metrics and metrics['converged_at_batch'] is not None:
            logger.info(f"{tally_type.capitalize()} converged at batch {metrics['converged_at_batch']}")
        else:
            logger.warning(f"{tally_type.capitalize()} did not converge within the simulated batches")
    
    return convergence_metrics

def plot_convergence(convergence_metrics, tally_type='dose'):
    """
    Plot the convergence of a specific tally type over batches.
    
    Parameters:
    -----------
    convergence_metrics : dict
        Dictionary with convergence metrics
    tally_type : str
        Type of tally to plot ('flux', 'dose', 'kerma', 'heating')
    
    Returns:
    --------
    fig : matplotlib.Figure
        Figure containing the convergence plot
    """
    if tally_type not in convergence_metrics:
        logger.warning(f"No convergence data found for '{tally_type}'")
        return None
    
    metrics = convergence_metrics[tally_type]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    batches = metrics['batches']
    values = metrics['values']
    errors = metrics['errors']
    
    # Plot values with error bars
    ax.errorbar(batches, values, yerr=[e * v for e, v in zip(errors, values)], 
                marker='o', linestyle='-', capsize=4, label=f'{tally_type.capitalize()} Value')
    
    # Add horizontal line for final value
    ax.axhline(metrics['final_value'], color='red', linestyle='--', 
               label=f'Final Value: {metrics["final_value"]:.4e}')
    
    # Add vertical line for convergence point
    if metrics['converged_at_batch'] is not None:
        ax.axvline(metrics['converged_at_batch'], color='green', linestyle='--', 
                   label=f'Converged at Batch {metrics["converged_at_batch"]}')
    
    # Add labels and title
    ax.set_xlabel('Batch Number')
    ax.set_ylabel(f'{tally_type.capitalize()} Value')
    ax.set_title(f'Convergence of {tally_type.capitalize()} over Batches')
    ax.legend()
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig
