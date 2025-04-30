#!/usr/bin/env python3
"""
Source definition for gamma-ray streaming simulation through concrete shield.
"""
import openmc
import numpy as np
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
        
        # Calculate solid angle subtended by the channel
        # For a cone with height h and base radius r:
        # solid_angle = 2π(1 - cos(θ)) where θ = tan^-1(r/h)
        h = SOURCE_TO_WALL_DISTANCE
        r = channel_radius
        theta = np.arctan(r / h)
        solid_angle = 2 * np.pi * (1 - np.cos(theta))
        total_angle = 4 * np.pi  # Full sphere
        
        # Calculate the fraction of particles that would hit the channel
        # This is the ratio of the solid angle to the total angle
        channel_fraction = solid_angle / total_angle
        
        logger.info(f"Channel subtends a solid angle of {solid_angle:.6f} sr")
        logger.info(f"Fraction of particles that would hit channel: {channel_fraction:.6f}")
        
        if biased:
            # Biased sampling to ensure particles are directed toward the channel
            # Define a conical distribution directed toward the channel
            cone_mu = np.cos(theta)
            
            # Define spatial distribution as point source
            spatial_dist = openmc.stats.Point(position)
            
            # Define angle distribution (biased toward channel)
            angle_dist = openmc.stats.Monodirectional((0, 0, 1))
            
            # Create source
            source = openmc.Source(space=spatial_dist, angle=angle_dist, energy=energy_dist)
            
            # Define the reference direction (toward the channel entrance)
            source.angle = openmc.stats.PolarAzimuthal(
                mu=openmc.stats.Uniform(cone_mu, 1.0),
                phi=openmc.stats.Uniform(0., 2*np.pi)
            )
            
            logger.info(f"Created biased point source with cone_mu={cone_mu:.6f}")
        else:
            # Isotropic source - not biased
            source = openmc.Source(space=openmc.stats.Point(position),
                                   angle=openmc.stats.Isotropic(),
                                   energy=energy_dist)
            logger.info("Created isotropic point source")
        
        return source
