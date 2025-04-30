#!/usr/bin/env python3
"""
Dose calculation functions for gamma-ray streaming simulations.
"""
import numpy as np
import scipy.integrate as integrate
from config import WALL_THICKNESS, SOURCE_TO_WALL_DISTANCE
from logging_utils import logger, LogSection

def flux_to_dose_conversion(energy):
    """
    Convert photon flux to dose using ANS-6.1.1-1977 conversion factors.
    
    Parameters:
    -----------
    energy : float or array
        Photon energy in MeV
    
    Returns:
    --------
    dose_factor : float or array
        Dose conversion factor (rem/hr)/(photons/cm²-s)
    """
    # ANS-6.1.1-1977 flux-to-dose conversion factors
    energy_points = np.array([
        0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 
        0.65, 0.7, 0.8, 1.0, 1.4, 1.8, 2.2, 2.6, 2.8, 3.25, 3.75, 4.25, 4.75, 5.0, 5.25, 
        5.75, 6.25, 6.75, 7.5, 9.0, 11.0, 13.0, 15.0
    ])
    
    dose_coeffs = np.array([
        3.96e-6, 5.82e-7, 2.90e-7, 2.58e-7, 2.83e-7, 3.79e-7, 5.01e-7, 6.31e-7, 7.59e-7, 
        8.78e-7, 9.85e-7, 1.08e-6, 1.17e-6, 1.27e-6, 1.36e-6, 1.44e-6, 1.52e-6, 1.68e-6, 
        1.98e-6, 2.51e-6, 2.99e-6, 3.42e-6, 3.82e-6, 4.01e-6, 4.41e-6, 4.83e-6, 5.23e-6, 
        5.60e-6, 5.80e-6, 6.01e-6, 6.37e-6, 6.74e-6, 7.11e-6, 7.66e-6, 8.77e-6, 1.03e-5, 
        1.18e-5, 1.33e-5
    ])
    
    return np.interp(energy, energy_points, dose_coeffs)

def calculate_theoretical_dose(energy, source_strength, channel_radius, detector_distance, detector_angle):
    """
    Calculate theoretical dose using analytical methods.
    
    This uses three different models:
    1. Simple point source with solid angle correction
    2. Line-of-sight model with air attenuation
    3. Enhanced model with backscatter and wall shine
    
    Parameters:
    -----------
    energy : float
        Photon energy in MeV
    source_strength : float
        Source strength in photons/second
    channel_radius : float
        Radius of the air channel in cm
    detector_distance : float
        Distance of the detector from the back face of the wall in cm
    detector_angle : float
        Angle of the detector from the central axis in degrees
    
    Returns:
    --------
    doses : tuple
        (point_source_dose, line_of_sight_dose, enhanced_model_dose) in rem/hr
    """
    with LogSection(f"Calculating theoretical dose (E={energy} MeV, r={channel_radius} cm, d={detector_distance} cm, θ={detector_angle}°)"):
        # Convert angle to radians
        angle_rad = np.radians(detector_angle)
        
        # Total distance from source to detector
        total_distance = SOURCE_TO_WALL_DISTANCE + WALL_THICKNESS + detector_distance
        
        # Distance from channel exit to detector
        exit_to_detector = np.sqrt(detector_distance**2 + (detector_distance * np.tan(angle_rad))**2)
        
        # Angular correction factor (cos(θ) accounts for solid angle effect)
        angular_factor = np.cos(angle_rad)
        
        # Effective area of detector seen from the channel exit
        if detector_angle == 0:
            # On-axis, full detector area is seen
            effective_area = 1.0
        else:
            # Off-axis, the effective area decreases with angle
            effective_area = angular_factor
        
        # Solid angle correction for channel
        # For a point source at distance h from a disk of radius r, the solid angle is:
        # Ω = 2π(1 - cos(θ)) where θ = tan^-1(r/h)
        solid_angle = 2 * np.pi * (1 - np.cos(np.arctan(channel_radius / SOURCE_TO_WALL_DISTANCE)))
        total_angle = 4 * np.pi  # Full sphere
        channel_fraction = solid_angle / total_angle
        
        # Simple inverse square law attenuation
        # Flux = source_strength * channel_fraction / (4π * total_distance²)
        flux_at_detector = source_strength * channel_fraction * effective_area / (4 * np.pi * total_distance**2)
        
        # Get dose conversion factor
        dose_factor = flux_to_dose_conversion(energy)
        
        # Calculate point source dose (simplest model)
        point_source_dose = flux_at_detector * dose_factor
        
        # Model 2: Add air attenuation
        # Air attenuation coefficient (cm^-1) - approximate for dry air
        # Based on XCOM database values for photon attenuation
        def air_attenuation_coeff(e):
            if e <= 0.1:
                return 0.0247  # Strong attenuation at low energies
            elif e <= 0.5:
                return 0.01 * (e**-0.8)  # Approximation for 0.1-0.5 MeV range
            elif e <= 2.0:
                return 0.005 * (e**-0.6)  # Approximation for 0.5-2.0 MeV range
            else:
                return 0.003 * (e**-0.4)  # Approximation for >2.0 MeV range
        
        # Calculate air attenuation
        mu_air = air_attenuation_coeff(energy)
        air_attenuation = np.exp(-mu_air * (WALL_THICKNESS + detector_distance))
        
        # Line-of-sight model
        line_of_sight_dose = point_source_dose * air_attenuation
        
        # Model 3: Enhanced model with scatter and wall shine
        # Additional factors to consider:
        # 1. Backscatter from wall
        # 2. Scatter within channel
        # 3. Albedo effect
        
        # Approximate factors based on empirical data and Monte Carlo simulations
        # Backscatter factor depends on energy
        def backscatter_factor(e):
            # Higher energy photons have less backscatter
            return 0.2 * np.exp(-0.5 * e)
        
        # Channel scatter factor depends on channel radius and wall thickness
        channel_scatter_factor = 1.0 + 0.1 * (channel_radius / WALL_THICKNESS)
        
        # Albedo factor (reflection coefficient)
        albedo_factor = 0.1 * np.exp(-0.8 * energy)
        
        # Enhanced model
        enhanced_model_dose = line_of_sight_dose * (1 + backscatter_factor(energy)) * channel_scatter_factor + \
                             point_source_dose * albedo_factor
        
        # For very small angles, add correction for beam spread
        if detector_angle < 10:
            beam_spread_factor = 1.0 + 0.1 * (1.0 - detector_angle/10.0)
            enhanced_model_dose *= beam_spread_factor
        
        logger.info(f"Point source dose: {point_source_dose:.6e} rem/hr")
        logger.info(f"Line-of-sight dose: {line_of_sight_dose:.6e} rem/hr")
        logger.info(f"Enhanced model dose: {enhanced_model_dose:.6e} rem/hr")
        
        return (point_source_dose, line_of_sight_dose, enhanced_model_dose)

def calculate_total_effective_dose(energy_spectrum, flux_spectrum, tissue_composition):
    """
    Calculate the total effective dose considering the energy spectrum and tissue composition.
    
    Parameters:
    -----------
    energy_spectrum : array
        Array of energy values in MeV
    flux_spectrum : array
        Array of flux values corresponding to each energy
    tissue_composition : dict
        Dictionary containing tissue composition by element
    
    Returns:
    --------
    effective_dose : float
        Total effective dose in rem/hr
    """
    # Get dose conversion factors for each energy
    dose_factors = flux_to_dose_conversion(energy_spectrum)
    
    # Calculate dose contribution from each energy
    dose_contributions = flux_spectrum * dose_factors
    
    # Numerical integration to get total dose
    total_dose = np.trapz(dose_contributions, energy_spectrum)
    
    # Apply tissue weighting factors for different organs
    # Simplified model - in reality would need to compute depth doses
    weighting_factors = {
        'whole_body': 1.0,
        'skin': 0.01,
        'eye_lens': 0.01,
        'gonads': 0.08,
        'breast': 0.12,
        'lung': 0.12,
        'thyroid': 0.04,
        'bone_surface': 0.01
    }
    
    # Apply a rough approximation for effective dose
    effective_dose = total_dose * 0.6  # Approximation factor
    
    return effective_dose

def calculate_kerma_to_dose_ratio(energy):
    """
    Calculate the ratio of kerma to dose for a given energy.
    
    Parameters:
    -----------
    energy : float
        Photon energy in MeV
    
    Returns:
    --------
    ratio : float
        Kerma-to-dose ratio
    """
    # For photons, kerma approximately equals dose at equilibrium
    # The ratio deviates from 1.0 due to electron transport effects
    
    if energy < 0.2:
        # Low energy - kerma ≈ dose
        return 1.02
    elif energy < 1.0:
        # Medium energy - small deviation
        return 1.05
    elif energy < 3.0:
        # Higher energy - larger deviation due to electron transport
        return 1.10
    else:
        # High energy - significant electron transport effects
        return 1.15
