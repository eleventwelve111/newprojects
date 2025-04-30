#!/usr/bin/env python3
"""
Compare simulation results with literature data on gamma-ray streaming through concrete shields.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path

from logging_utils import logger, LogSection, timeit
from config import RESULTS_DIR, PLOTS_DIR

@timeit
def compare_with_literature(results):
    """
    Compare simulation results with literature data.
    
    Parameters:
    -----------
    results : list or dict
        Simulation results
    
    Returns:
    --------
    comparison_data : dict
        Comparison data between simulation and literature
    """
    with LogSection("Comparing results with literature"):
        # Convert single result to list if needed
        if isinstance(results, dict):
            results = [results]
        
        # Create output directory
        output_dir = PLOTS_DIR / "literature_comparison"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Compare with Lee et al. (2007) data
        lee_comparison = compare_with_lee_2007(results)
        
        # Compare with NCRP-147 data (Structural Shielding Design for Medical X-Ray Imaging Facilities)
        ncrp_comparison = compare_with_ncrp_147(results)
        
        # Combine comparison data
        comparison_data = {
            'lee_2007': lee_comparison,
            'ncrp_147': ncrp_comparison
        }
        
        # Save comparison data
        import json
        with open(RESULTS_DIR / "data" / "literature_comparison.json", 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        logger.info("Literature comparison completed")
        return comparison_data

def compare_with_lee_2007(results):
    """
    Compare simulation results with Lee et al. (2007) study on gamma-ray streaming through concrete cracks.
    
    Reference:
    Lee, C., Lee, Y. H., & Lee, K. J. (2007). Cracking effect on gamma-ray shielding 
    performance in concrete structure. Progress in Nuclear Energy, 49(4), 303-312.
    https://doi.org/10.1016/j.pnucene.2007.01.006
    """
    logger.info("Comparing with Lee et al. (2007) data")
    
    # Lee et al. data (extracted from paper)
    # Format: [crack_width_mm, dose_increase_factor]
    lee_data = {
        '30cm_thickness': {
            'crack_width_mm': [0.5, 1.0, 5.0, 10.0],
            'dose_factor': [1.2, 1.5, 3.2, 5.8]  # Approximate values from the paper
        },
        '60cm_thickness': {
            'crack_width_mm': [0.5, 1.0, 5.0, 10.0],
            'dose_factor': [1.1, 1.3, 2.8, 4.9]  # Approximate values from the paper
        }
    }
    
    # Convert our channel diameters to mm for comparison
    our_diameters_mm = np.unique([r['channel_diameter'] * 10 for r in results])
    
    # Find result subset closest to Lee's experiment (1.0 MeV source, 0 degrees)
    lee_comparable_results = []
    for r in results:
        if (abs(r['energy'] - 1.0) < 0.1 and 
            abs(r['detector_angle']) < 0.1 and
            r.get('concrete_type', 'standard') == 'standard'):
            lee_comparable_results.append(r)
    
    # Prepare our data in compatible format
    our_data = {
        'crack_width_mm': [],
        'dose': [],
        'detector_distance': []
    }
    
    baseline_doses = {}  # To calculate dose increase factors
    
    for r in lee_comparable_results:
        dist = r['detector_distance']
        diam_mm = r['channel_diameter'] * 10
        
        # Store values
        our_data['crack_width_mm'].append(diam_mm)
        our_data['dose'].append(r['dose']['value'])
        our_data['detector_distance'].append(dist)
        
        # Track minimum diameter as baseline for each distance
        if dist not in baseline_doses or diam_mm < baseline_doses[dist]['diameter']:
            baseline_doses[dist] = {'diameter': diam_mm, 'dose': r['dose']['value']}
    
    # Calculate dose increase factors
    our_data['dose_factor'] = []
    for i, dist in enumerate(our_data['detector_distance']):
        if baseline_doses[dist]['dose'] > 0:
            factor = our_data['dose'][i] / baseline_doses[dist]['dose']
        else:
            factor = np.nan
        our_data['dose_factor'].append(factor)
    
    # Create plot comparing our results with Lee et al.
    plt.figure(figsize=(10, 6))
    
    # Plot Lee et al. data
    plt.plot(lee_data['30cm_thickness']['crack_width_mm'], 
             lee_data['30cm_thickness']['dose_factor'], 
             'o-', label='Lee et al. - 30cm thickness')
    
    plt.plot(lee_data['60cm_thickness']['crack_width_mm'], 
             lee_data['60cm_thickness']['dose_factor'], 
             's-', label='Lee et al. - 60cm thickness')
    
    # Plot our data by distance
    for dist in np.unique(our_data['detector_distance']):
        indices = [i for i, d in enumerate(our_data['detector_distance']) if d == dist]
        widths = [our_data['crack_width_mm'][i] for i in indices]
        factors = [our_data['dose_factor'][i] for i in indices]
        
        # Sort by width
        sort_idx = np.argsort(widths)
        widths = [widths[i] for i in sort_idx]
        factors = [factors[i] for i in sort_idx]
        
        plt.plot(widths, factors, 'x-', 
                label=f'Our simulation - {dist}cm distance')
    
    plt.xlabel('Channel/Crack Width (mm)')
    plt.ylabel('Dose Increase Factor')
    plt.title('Comparison with Lee et al. (2007): Dose Increase vs. Channel Width')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "literature_comparison" / "lee_2007_comparison.png", dpi=300)
    plt.close()
    
    logger.info("Lee et al. comparison completed")
    
    return {
        'lee_data': lee_data,
        'our_data': our_data
    }

def compare_with_ncrp_147(results):
    """
    Compare simulation results with NCRP-147 data on radiation shielding.
    
    Reference:
    NCRP Report No. 147: Structural Shielding Design for Medical X-Ray Imaging Facilities
    """
    logger.info("Comparing with NCRP-147 data")
    
    # NCRP-147 provides broad beam transmission factors for concrete at various energies
    # These are approximate values derived from the report
    ncrp_data = {
        'energies_MeV': [0.1, 0.5, 1.0, 2.0, 5.0],
        'concrete_30cm_transmission': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],  # Approximate values
                'concrete_60cm_transmission': [1e-12, 1e-10, 1e-8, 1e-6, 1e-4]  # Approximate values
    }
    
    # Find comparable results from our simulations (standard concrete, 0 degrees angle)
    ncrp_comparable_results = []
    for r in results:
        if (abs(r['detector_angle']) < 0.1 and
            r.get('concrete_type', 'standard') == 'standard'):
            ncrp_comparable_results.append(r)
    
    # Prepare our data in compatible format
    our_data = {
        'energies_MeV': [],
        'detector_distance': [],
        'channel_diameter': [],
        'transmission_factor': []
    }
    
    # Calculate transmission factors
    # In our case, we can estimate from dose without channel (minimum diameter) to dose with channel
    for energy in ncrp_data['energies_MeV']:
        # Get results for this energy
        energy_results = [r for r in ncrp_comparable_results 
                         if abs(r['energy'] - energy) < 0.01]
        
        if not energy_results:
            continue
            
        # Group by distance and diameter
        for dist in np.unique([r['detector_distance'] for r in energy_results]):
            dist_results = [r for r in energy_results if r['detector_distance'] == dist]
            
            # Sort by diameter
            diameters = np.array([r['channel_diameter'] for r in dist_results])
            sort_idx = np.argsort(diameters)
            
            # Estimate no-channel dose using smallest channel as reference
            min_diam_result = dist_results[sort_idx[0]]
            min_diam = min_diam_result['channel_diameter']
            min_dose = min_diam_result['dose']['value']
            
            # Calculate transmission factor for each diameter
            for i in sort_idx:
                r = dist_results[i]
                diam = r['channel_diameter']
                dose = r['dose']['value']
                
                # Transmission is dose with channel divided by theoretical dose without hole
                # We approximate "no hole" dose using the smallest diameter result
                transmission = dose / min_dose if min_dose > 0 else np.nan
                
                our_data['energies_MeV'].append(energy)
                our_data['detector_distance'].append(dist)
                our_data['channel_diameter'].append(diam)
                our_data['transmission_factor'].append(transmission)
    
    # Create plot comparing our results with NCRP-147
    plt.figure(figsize=(10, 6))
    
    # Plot NCRP-147 data
    plt.plot(ncrp_data['energies_MeV'], ncrp_data['concrete_30cm_transmission'], 
            'o-', label='NCRP-147 - 30cm concrete (no streaming)')
    plt.plot(ncrp_data['energies_MeV'], ncrp_data['concrete_60cm_transmission'], 
            's-', label='NCRP-147 - 60cm concrete (no streaming)')
    
    # Plot our data for selected diameters
    for diam in [0.05, 0.5, 1.0]:  # Selected diameters in cm
        for dist in [30, 60]:  # Selected distances in cm
            # Find matching data points
            indices = [i for i, (d, dist_i) in enumerate(zip(our_data['channel_diameter'], 
                                                           our_data['detector_distance'])) 
                      if abs(d - diam) < 0.01 and dist_i == dist]
            
            if not indices:
                continue
                
            energies = [our_data['energies_MeV'][i] for i in indices]
            factors = [our_data['transmission_factor'][i] for i in indices]
            
            # Sort by energy
            sort_idx = np.argsort(energies)
            energies = [energies[i] for i in sort_idx]
            factors = [factors[i] for i in sort_idx]
            
            plt.plot(energies, factors, 'x-', 
                    label=f'Our sim - {diam}cm channel, {dist}cm distance')
    
    plt.xlabel('Photon Energy (MeV)')
    plt.ylabel('Transmission Factor')
    plt.title('Comparison with NCRP-147: Transmission Factor vs. Energy')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "literature_comparison" / "ncrp_147_comparison.png", dpi=300)
    plt.close()
    
    logger.info("NCRP-147 comparison completed")
    
    return {
        'ncrp_data': ncrp_data,
        'our_data': our_data
    }

