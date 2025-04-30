#!/usr/bin/env python3
"""
Source definition for gamma-ray streaming simulation through concrete shield.
"""
import openmc
import numpy as np
import math
from config import SOURCE_TO_WALL_DISTANCE, WALL_THICKNESS
from logging_utils import logger, LogSection

def create_point_source(energy, channel_radius, biased=True):
    """
    Create a gamma-ray point source directed toward the air channel.
    
    Parameters:
    -----------
    energy : float
        Gamma-ray energy in MeV
    channel_radius : float
        Radius of the air channel in cm
    biased : bool
        Whether to bias the source toward the channel
    
    Returns:
    --------
    source : openmc.Source
        OpenMC source
    """
    with LogSection(f"Creating point source (E={energy} MeV, channel radius={channel_radius} cm)"):
        # Source position (centered along the z-axis at -SOURCE_TO_WALL_DISTANCE)
        position = (0, 0, -SOURCE_TO_WALL_DISTANCE)
        
        # Define energy distribution (monoenergetic)
        energy_dist = openmc.stats.Discrete([energy], [1.0])
        
        # Calculate the solid angle subtended by the channel
        # from the source position
        theta_max = np.arctan(channel_radius / SOURCE_TO_WALL_DISTANCE)
        
        if biased:
            # Create a biased angular distribution to focus toward the channel
            # without biasing physics once particles are sampled
            mu_min = np.cos(theta_max)
            
            # Create a custom angular distribution that focuses towards the channel
            # This ensures 100% of particles go through the channel
            def custom_pdf(theta, phi):
                # Only sample directions within cone to channel
                if np.cos(theta) >= mu_min:
                    return 1.0
                else:
                    return 0.0
                    
            # Create the biased angular distribution
            angle_dist = openmc.stats.PolarAzimuthal(
                mu=openmc.stats.CustomDiscrete(
                    np.linspace(mu_min, 1.0, 20),
                    np.ones(20) / 20
                ),
                phi=openmc.stats.Uniform(0, 2*np.pi)
            )
            
            logger.info(f"Source biased toward channel with max angle: {np.degrees(theta_max):.4f} degrees")
            logger.info(f"Fraction of isotropic source directed at channel: {(1-mu_min)/2:.6e}")
        else:
            # Use isotropic emission for unbiased source
            angle_dist = openmc.stats.Isotropic()
            logger.info("Using isotropic source (no bias)")
        
        # Create source
        source = openmc.Source(space=openmc.stats.Point(position),
                              angle=angle_dist,
                              energy=energy_dist)
        
        return source
