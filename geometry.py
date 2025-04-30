#!/usr/bin/env python3
"""
Geometry setup for gamma-ray streaming simulation through concrete shield.
"""
import openmc
import numpy as np
import math
from config import WALL_THICKNESS, SOURCE_TO_WALL_DISTANCE, DETECTOR_DIAMETER
from logging_utils import logger, LogSection

def create_geometry(concrete, air, tissue, channel_diameter, detector_distance, detector_angle, concrete_type="standard"):
    """
    Create geometry for the simulation with a concrete wall, air channel, and ICRU sphere.
    
    Parameters:
    -----------
    concrete : openmc.Material
        Concrete material
    air : openmc.Material
        Air material
    tissue : openmc.Material
        Tissue material (ICRU sphere)
    channel_diameter : float
        Diameter of the air channel in cm
    detector_distance : float
        Distance of the detector from the back face of the wall in cm
    detector_angle : float
        Angle of the detector from the central axis in degrees
    concrete_type : str
        Type of concrete to use (standard, barite, magnetite)
        
    Returns:
    --------
    geometry : openmc.Geometry
        OpenMC geometry
    """
    with LogSection(f"Creating geometry (channel Ø={channel_diameter} cm, dist={detector_distance} cm, ang={detector_angle}°)"):
        # Dimensions
        channel_radius = channel_diameter / 2.0
        
        # Define boundary box dimensions (make it large enough to contain everything)
        boundary_box_width = 500.0  # cm
        boundary_box_height = 500.0  # cm
        boundary_box_depth = SOURCE_TO_WALL_DISTANCE + WALL_THICKNESS + detector_distance * 2  # cm
        
        # Create concrete wall
        wall_min_x = boundary_box_width / -2
        wall_max_x = boundary_box_width / 2
        wall_min_y = boundary_box_height / -2
        wall_max_y = boundary_box_height / 2
        wall_min_z = 0.0
        wall_max_z = WALL_THICKNESS
        
        wall_region = +openmc.ZPlane(wall_min_z) & -openmc.ZPlane(wall_max_z) & \
                      +openmc.XPlane(wall_min_x) & -openmc.XPlane(wall_max_x) & \
                      +openmc.YPlane(wall_min_y) & -openmc.YPlane(wall_max_y)
        
        # Create air channel through wall (cylindrical)
        channel_axis = openmc.ZCylinder(x0=0, y0=0, r=channel_radius)
        channel_region = channel_axis & +openmc.ZPlane(wall_min_z) & -openmc.ZPlane(wall_max_z)
        
        # Create concrete wall with channel
        wall_with_channel_region = wall_region & ~channel_region
        wall_with_channel_cell = openmc.Cell(name='concrete_wall')
        wall_with_channel_cell.region = wall_with_channel_region
        wall_with_channel_cell.fill = concrete
        
        # Create channel cell
        channel_cell = openmc.Cell(name='air_channel')
        channel_cell.region = channel_region
        channel_cell.fill = air
        
        # Create ICRU sphere at specified distance and angle
        # First calculate the position
        sphere_z = WALL_THICKNESS + detector_distance * np.cos(np.radians(detector_angle))
        sphere_x = detector_distance * np.sin(np.radians(detector_angle))
        sphere_y = 0.0
        
        detector_sphere = openmc.Sphere(x0=sphere_x, y0=sphere_y, z0=sphere_z, r=DETECTOR_DIAMETER/2)
        detector_cell = openmc.Cell(name='detector')
        detector_cell.region = -detector_sphere
        detector_cell.fill = tissue
        
        # Create void cell for the external environment
        box_region = +openmc.XPlane(wall_min_x) & -openmc.XPlane(wall_max_x) & \
                    +openmc.YPlane(wall_min_y) & -openmc.YPlane(wall_max_y) & \
                    +openmc.ZPlane(-SOURCE_TO_WALL_DISTANCE) & -openmc.ZPlane(wall_max_z + 300)
        
        # Void region is everything else inside the boundary box
        void_region = box_region & ~wall_with_channel_region & ~channel_region & ~detector_cell.region
        void_cell = openmc.Cell(name='void')
        void_cell.region = void_region
        
        # Set boundary conditions
        for surface in [openmc.XPlane(wall_min_x), openmc.XPlane(wall_max_x),
                        openmc.YPlane(wall_min_y), openmc.YPlane(wall_max_y),
                        openmc.ZPlane(-SOURCE_TO_WALL_DISTANCE), openmc.ZPlane(wall_max_z + 300)]:
            surface.boundary_type = 'vacuum'
        
        # Create geometry
        geometry = openmc.Geometry([wall_with_channel_cell, channel_cell, detector_cell, void_cell])
        
        # Export to XML for visualization and verification
        geometry.export_to_xml()
        
        # Log the geometry details
        logger.info(f"Created concrete wall with thickness {WALL_THICKNESS} cm")
        logger.info(f"Created air channel with diameter {channel_diameter} cm")
        logger.info(f"Created ICRU sphere at distance {detector_distance} cm and angle {detector_angle}°")
        
        return geometry
