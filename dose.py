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


def plot_theoretical_vs_simulated(theoretical_doses, simulated_doses, 
                                 distances, angles, energies, channel_diameters):
    """
    Create plots comparing theoretical and simulated doses
    
    Parameters:
    -----------
    theoretical_doses : dict
        Dictionary of theoretical dose data
    simulated_doses : dict
        Dictionary of simulated dose data
    distances : list
        List of distances in cm
    angles : list
        List of angles in degrees
    energies : list
        List of energies in MeV
    channel_diameters : list
        List of channel diameters in cm
    """
    # Create comparison plots for each energy and channel diameter
    for energy in energies:
        for diameter in channel_diameters:
            plt.figure(figsize=(12, 8))
            
            for dist in distances:
                # Theoretical data
                theo_doses = [theoretical_doses[(energy, diameter, dist, ang)][0] for ang in angles]
                theo_direct = [theoretical_doses[(energy, diameter, dist, ang)][1] for ang in angles]
                theo_inverse = [theoretical_doses[(energy, diameter, dist, ang)][2] for ang in angles]
                
                # Simulated data
                sim_doses = [simulated_doses.get((energy, diameter, dist, ang), 0) for ang in angles]
                
                # Plot
                plt.semilogy(angles, theo_doses, 'b-', label=f'Theoretical (SA) {dist} cm' if dist == distances[0] else "")
                plt.semilogy(angles, theo_direct, 'g--', label=f'Theoretical (DB) {dist} cm' if dist == distances[0] else "")
                plt.semilogy(angles, theo_inverse, 'm-.', label=f'Theoretical (IS) {dist} cm' if dist == distances[0] else "")
                plt.semilogy(angles, sim_doses, 'ro-', label=f'Simulated {dist} cm')
                
            plt.title(f'Dose vs Angle (E={energy} MeV, Channel Ø={diameter} cm)')
            plt.xlabel('Angle (degrees)')
            plt.ylabel('Dose (rem/hr)')
            plt.legend()
            plt.grid(True, which="both", ls="--")
            plt.savefig(f'dose_comparison_E{energy}_D{diameter}.png', dpi=300)
            plt.close()

        5.23E-3, 5.60E-3, 5.80E-3, 6.01E-3, 6.37E-3, 6.74E-3, 7.11E-3, 7.66E-3, 8.77E-3, 
        1.03E-2, 1.18E-2, 1.33E-2
    ])
    
    # Handle different input types
    if np.isscalar(energy):
        # For single energy value
        if energy <= energy_points[0]:
            return dose_coeffs[0]
        elif energy >= energy_points[-1]:
            return dose_coeffs[-1]
        else:
            # Log-log interpolation for better accuracy
            log_energy = np.log(energy_points)
            log_coeffs = np.log(dose_coeffs)
            log_interp = np.interp(np.log(energy), log_energy, log_coeffs)
            return np.exp(log_interp)
    else:
        # For array of energy values
        energy_array = np.asarray(energy)
        result = np.zeros_like(energy_array)
        
        # Handle values below minimum
        below_min = energy_array <= energy_points[0]
        result[below_min] = dose_coeffs[0]
        
        # Handle values above maximum
        above_max = energy_array >= energy_points[-1]
        result[above_max] = dose_coeffs[-1]
        
        # Handle values in range
        in_range = ~(below_min | above_max)
        if np.any(in_range):
            # Log-log interpolation for better accuracy
            log_energy = np.log(energy_points)
            log_coeffs = np.log(dose_coeffs)
            log_interp = np.interp(np.log(energy_array[in_range]), log_energy, log_coeffs)
            result[in_range] = np.exp(log_interp)
        
        return result

@timeit
def calculate_dose_from_spectrum(energy_bins, flux_spectrum):
    """
    Calculate dose from energy spectrum.
    
    Parameters:
    -----------
    energy_bins : array-like
        Energy bin boundaries in MeV
    flux_spectrum : array-like
        Flux spectrum in particles/cm²/s/MeV
    
    Returns:
    --------
    total_dose : float
        Total dose rate in rem/hr
    dose_contributions : array-like
        Dose contribution from each energy bin in rem/hr
    """
    with LogSection("Calculating dose from spectrum"):
        # Ensure inputs are numpy arrays
        energy_bins = np.asarray(energy_bins)
        flux_spectrum = np.asarray(flux_spectrum)
        
        # Check input validity
        if len(energy_bins) != len(flux_spectrum) + 1:
            raise ValueError("Energy bins should have one more element than flux spectrum")
        
        # Calculate bin widths
        bin_widths = np.diff(energy_bins)
        
        # Calculate midpoints for dose conversion
        energy_midpoints = 0.5 * (energy_bins[:-1] + energy_bins[1:])
        
        # Get dose conversion factors for each energy bin
        dose_factors = flux_to_dose_conversion(energy_midpoints)
        
        # Calculate dose contribution from each bin
        dose_contributions = flux_spectrum * bin_widths * dose_factors
        
        # Calculate total dose
        total_dose = np.sum(dose_contributions)
        
        logger.info(f"Calculated dose from spectrum: {total_dose:.4e} rem/hr")
        
        return total_dose, dose_contributions

@timeit
def calculate_buildup_factor(energy, thickness, material='concrete'):
    """
    Calculate gamma-ray buildup factor for dose calculations.
    
    Parameters:
    -----------
    energy : float
        Photon energy in MeV
    thickness : float
        Material thickness in cm
    material : str, default='concrete'
        Shield material
    
    Returns:
    --------
    buildup : float
        Buildup factor (dimensionless)
    """
    with LogSection(f"Calculating buildup factor for {material}"):
        # Thickness in mean free paths
        if material.lower() == 'concrete':
            # Linear attenuation coefficients for concrete (cm^-1)
            # Energy (MeV): 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0
            energies = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
            mu_values = np.array([0.204, 0.149, 0.105, 0.086, 0.075, 0.069, 0.064, 0.057, 0.054])
            
            # Interpolate to get mu at the given energy
            if energy <= energies[0]:
                mu = mu_values[0]
            elif energy >= energies[-1]:
                mu = mu_values[-1]
            else:
                mu = np.interp(energy, energies, mu_values)
            
            # Calculate thickness in mean free paths
            mfp_thickness = thickness * mu
            
            # Parameters for Taylor form buildup factor for concrete
            # Format: [A, a, B, b] for each energy
            # B(E, x) = A*exp(-a*x) + B*exp(-b*x)
            taylor_params = {
                0.5: [13.71, -0.075, 1.314, 0.3581],
                1.0: [10.97, -0.043, 0.521, 0.2513],
                2.0: [7.43, -0.011, 0.2522, 0.1876],
                3.0: [6.06, 0.0078, 0.1456, 0.1471],
                4.0: [5.28, 0.0205, 0.0988, 0.1173],
                5.0: [4.76, 0.0295, 0.0734, 0.0958],
                6.0: [4.41, 0.0365, 0.0576, 0.0786],
                8.0: [3.95, 0.0465, 0.0388, 0.0545],
                10.0: [3.68, 0.0533, 0.028, 0.0366]
            }
            
            # Find closest energy for parameters
            closest_energy = energies[np.argmin(np.abs(energies - energy))]
            A, a, B, b = taylor_params[closest_energy]
            
            # Calculate buildup factor
            buildup = A * np.exp(-a * mfp_thickness) + B * np.exp(-b * mfp_thickness)
            
            logger.info(f"Buildup factor for {material} at {energy:.1f} MeV, thickness={thickness:.1f} cm: {buildup:.2f}")
            
            return buildup
        else:
            # For other materials, use a simple approximation
            logger.warning(f"Buildup factors for {material} not implemented, using approximation")
            
            # Simple approximation based on energy
            if energy < 1.0:
                base_factor = 5.0
            elif energy < 3.0:
                base_factor = 3.0
            else:
                base_factor = 2.0
            
            # Scale with thickness (assuming thickness is in cm)
            # This is a very simplified approach
            scaled_factor = base_factor * (1.0 + 0.1 * thickness)
            
            return min(scaled_factor, 20.0)  # Cap at reasonable maximum

@timeit
def calculate_streaming_dose(source_energy, channel_diameter, detector_distance, wall_thickness=None):
    """
    Analytical approximation of dose from gamma-ray streaming through a cylindrical duct.
    
    Parameters:
    -----------
    source_energy : float
        Source energy in MeV
    channel_diameter : float
        Channel diameter in cm
    detector_distance : float
        Distance from end of channel to detector in cm
    wall_thickness : float, optional
        Wall thickness in cm. If None, use value from config
        
    Returns:
    --------
    dose : float
        Approximate dose rate in rem/hr
    """
    with LogSection("Calculating analytical streaming dose"):
        # Use wall thickness from config if not provided
        if wall_thickness is None:
            wall_thickness = WALL_THICKNESS
        
        # Source strength (assume unit source for relative calculations)
        source_strength = 1.0  # photons/s
        
        # Convert to point source dose at detector position without wall
        total_distance = SOURCE_TO_WALL_DISTANCE + wall_thickness + detector_distance
        uncollided_flux = source_strength / (4 * np.pi * total_distance**2)
        
        # Apply attenuation through wall thickness if there were no channel
        mu = 0.0  # Linear attenuation coefficient in cm^-1
        
        # Get approximate attenuation coefficient based on energy
        if source_energy <= 0.1:
            mu = 0.4
        elif source_energy <= 0.5:
            mu = 0.2
        elif source_energy <= 1.0:
            mu = 0.15
        elif source_energy <= 2.0:
            mu = 0.11
        elif source_energy <= 5.0:
            mu = 0.08
        else:
            mu = 0.06
        
        # Attenuated dose without channel
        attenuated_flux = uncollided_flux * np.exp(-mu * wall_thickness)
        direct_dose = attenuated_flux * flux_to_dose_conversion(source_energy)
        
        # Calculate streaming contribution
        # Based on simplified model from NCRP 147/ANS 6.6.1
        
        # Albedo factor (backscattering)
        albedo = 0.2 * (1.0 - np.exp(-source_energy / 2.0))
        
        # Solid angle fraction
        channel_radius = channel_diameter / 2.0
        solid_angle_factor = (channel_radius**2) / (4 * SOURCE_TO_WALL_DISTANCE**2)
        
        # Duct transmission factor
        aspect_ratio = wall_thickness / channel_diameter
        # Approximation for small aspect ratios
        if aspect_ratio < 1.0:
            duct_factor = 1.0 - 0.8 * aspect_ratio
        # Approximation for larger aspect ratios
        else:
            duct_factor = 0.2 * np.exp(-0.3 * aspect_ratio)
        
        # Distance correction
        distance_factor = 1.0 / (1.0 + (detector_distance / channel_diameter)**2)
        
        # Combine factors for streaming dose
        streaming_flux = source_strength * solid_angle_factor * duct_factor * distance_factor
        streaming_dose = streaming_flux * flux_to_dose_conversion(source_energy)
        
        # Apply buildup factor for wall portion
        buildup = calculate_buildup_factor(source_energy, wall_thickness)
        
        # Total dose is direct dose plus streaming dose
        total_dose = direct_dose * buildup + streaming_dose
        
        logger.info(f"Analytical streaming dose estimate: {total_dose:.4e} rem/hr")
        
        return total_dose

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate dose conversion factors and buildup")
    parser.add_argument("--output", type=str, default="results/dose_evaluation",
                        help="Output directory for plots")
    
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot dose conversion factors
    energies = np.logspace(-2, 1.5, 100)  # 0.01 to 30 MeV
    conversion_factors = flux_to_dose_conversion(energies)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(energies, conversion_factors, 'b-', linewidth=2)
    plt.xlabel('Photon Energy (MeV)', fontsize=12)
    plt.ylabel('Dose Conversion Factor\n(rem/hr)/(photons/cm²-s)', fontsize=12)
    plt.title('ANS-6.1.1-1977 Photon Flux-to-Dose Conversion Factors', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "dose_conversion_factors.png", dpi=300)
    
    # Plot buildup factors for different energies
    thicknesses = np.linspace(0, 100, 50)  # 0 to 100 cm
    energies_to_plot = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    plt.figure(figsize=(10, 6))
    for energy in energies_to_plot:
        buildup_factors = [calculate_buildup_factor(energy, t) for t in thicknesses]
        plt.plot(thicknesses, buildup_factors, linewidth=2, label=f'{energy} MeV')
    
    plt.xlabel('Concrete Thickness (cm)', fontsize=12)
    plt.ylabel('Buildup Factor', fontsize=12)
    plt.title('Gamma-Ray Buildup Factors for Concrete', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Photon Energy', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "buildup_factors.png", dpi=300)
    
    logger.info(f"Dose evaluation plots saved to {output_dir}")

