#!/usr/bin/env python3
"""
Geometry setup for gamma-ray streaming simulation through concrete shield.
"""
import openmc
import numpy as np
import math
from config import WALL_THICKNESS, SOURCE_TO_WALL_DISTANCE, DETECTOR_DIAMETER
from logging_utils import logger, LogSection

def create_geometry(materials_dict, channel_diameter, detector_distance, detector_angle, concrete_type="standard"):
    """
    Create geometry for the simulation with a concrete wall, air channel, and ICRU sphere.
    
    Parameters:
    -----------
    materials_dict : dict
        Dictionary of materials
    channel_diameter : float
        Diameter of the air channel in cm
    detector_distance : float
        Distance of the detector from the back face of the wall in cm
    detector_angle : float
        Angle of the detector from the central axis in degrees
    concrete_type : str
        Type of concrete to use ("standard", "barite", or "magnetite")
    
    Returns:
    --------
    geometry : openmc.Geometry
        OpenMC geometry
    cells : dict
        Dictionary of cells
    """
    with LogSection(f"Creating geometry (channel Ø={channel_diameter} cm, detector at {detector_distance} cm, {detector_angle}°)"):
        # Get materials
        if concrete_type == "barite":
            concrete = materials_dict['barite_concrete']
            logger.info("Using barite concrete (high-density)")
        elif concrete_type == "magnetite":
            concrete = materials_dict['magnetite_concrete']
            logger.info("Using magnetite concrete (high-density)")
        else:
            concrete = materials_dict['standard_concrete']
            logger.info("Using standard concrete")
            
        air = materials_dict['air']
        tissue = materials_dict['tissue']
        void = materials_dict['void']
        
        # Create region for world
        world_min = -200
        world_max = 400
        world_box = openmc.model.rectangular_prism(
            width=world_max-world_min, 
            height=world_max-world_min, 
            axis='z',
            origin=(0, 0, (world_max+world_min)/2),
            boundary_type='vacuum'
        )
        
        # Wall region
        wall_min_z = 0
        wall_max_z = WALL_THICKNESS
        wall_region = openmc.model.rectangular_prism(
            width=200, 
            height=200, 
            axis='z',
            origin=(0, 0, WALL_THICKNESS/2),
            boundary_type='transmission'
        )
        wall_region = wall_region & openmc.ZPlane(z0=wall_min_z) & -openmc.ZPlane(z0=wall_max_z)
        
        # Air channel (cylindrical)
        channel_radius = channel_diameter / 2.0
        channel_region = openmc.ZCylinder(r=channel_radius)
        channel_region = channel_region & openmc.ZPlane(z0=wall_min_z) & -openmc.ZPlane(z0=wall_max_z)
        
        # Define the concrete wall cell (with the air channel subtracted)
        concrete_cell = openmc.Cell(name='Concrete Wall')
        concrete_cell.region = wall_region & ~channel_region
        concrete_cell.fill = concrete
        
        # Define the air channel cell
        air_channel_cell = openmc.Cell(name='Air Channel')
        air_channel_cell.region = channel_region
        air_channel_cell.fill = air
        
        # Define detector (ICRU sphere)
        detector_radius = DETECTOR_DIAMETER / 2.0
        
        # Calculate detector position based on distance and angle
        angle_rad = np.radians(detector_angle)
        detector_z = wall_max_z + detector_distance * np.cos(angle_rad)
        detector_x = detector_distance * np.sin(angle_rad)
        detector_y = 0.0
        
        detector_region = openmc.Sphere(x0=detector_x, y0=detector_y, z0=detector_z, r=detector_radius)
        detector_cell = openmc.Cell(name='ICRU Detector')
        detector_cell.region = detector_region
        detector_cell.fill = tissue
        
        # Source region (before wall)
        source_region = -openmc.ZPlane(z0=wall_min_z)
        source_cell = openmc.Cell(name='Source Region')
        source_cell.region = source_region & world_box
        source_cell.fill = void
        
        # External void region (after wall)
        external_region = +openmc.ZPlane(z0=wall_max_z)
        external_cell = openmc.Cell(name='External Void')
        external_cell.region = external_region & world_box & ~detector_region
        external_cell.fill = void
        
        # Create universe and geometry
        universe = openmc.Universe(cells=[concrete_cell, air_channel_cell, 
                                         detector_cell, source_cell, external_cell])
        geometry = openmc.Geometry(universe)
        
        # Create dictionary of cells for tallies
        cells = {
            'concrete': concrete_cell,
            'channel': air_channel_cell,
            'detector': detector_cell,
            'source_region': source_cell,
            'external': external_cell
        }
        
        logger.info(f"Created geometry with concrete shield and {channel_diameter} cm diameter air channel")
        logger.info(f"Detector positioned at ({detector_x:.2f}, {detector_y:.2f}, {detector_z:.2f}) cm")
        
        return geometry, cells
