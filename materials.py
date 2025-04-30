#!/usr/bin/env python3
"""
Materials for gamma-ray streaming simulation through concrete shield.
Based on ANSI/ANS-6.4-2006 for concrete compositions.
"""
import openmc
import numpy as np
from logging_utils import logger, LogSection

def create_materials():
    """
    Create materials for the simulation:
    - Standard concrete (ANSI/ANS-6.4-2006)
    - Barite concrete (high-density)
    - Magnetite concrete (high-density)
    - Air
    - Tissue (ICRU sphere)
    - Void
    
    Returns:
        materials: OpenMC Materials collection
        concrete: Standard concrete material
        barite_concrete: Barite concrete material
        magnetite_concrete: Magnetite concrete material
        air: Air material
        tissue: Tissue material
    """
    with LogSection("Creating materials"):
        # Create a material collection
        materials = openmc.Materials()
        
        # Standard concrete (ANSI/ANS-6.4-2006)
        concrete = openmc.Material(name='Standard Concrete')
        concrete.set_density('g/cm3', 2.3)
        concrete.add_element('H', 0.01, 'wo')
        concrete.add_element('O', 0.53, 'wo')
        concrete.add_element('Na', 0.016, 'wo')
        concrete.add_element('Mg', 0.002, 'wo')
        concrete.add_element('Al', 0.034, 'wo')
        concrete.add_element('Si', 0.337, 'wo')
        concrete.add_element('S', 0.002, 'wo')
        concrete.add_element('K', 0.013, 'wo')
        concrete.add_element('Ca', 0.044, 'wo')
        concrete.add_element('Fe', 0.014, 'wo')
        materials.append(concrete)
        logger.info(f"Created standard concrete material: {concrete}")
        
        # Barite concrete (high-density)
        barite_concrete = openmc.Material(name='Barite Concrete')
        barite_concrete.set_density('g/cm3', 3.35)
        barite_concrete.add_element('H', 0.003, 'wo')
        barite_concrete.add_element('O', 0.307, 'wo')
        barite_concrete.add_element('Mg', 0.001, 'wo')
        barite_concrete.add_element('Al', 0.008, 'wo')
        barite_concrete.add_element('Si', 0.01, 'wo')
        barite_concrete.add_element('S', 0.107, 'wo')
        barite_concrete.add_element('Ca', 0.05, 'wo')
        barite_concrete.add_element('Fe', 0.047, 'wo')
        barite_concrete.add_element('Ba', 0.467, 'wo')
        materials.append(barite_concrete)
        logger.info(f"Created barite concrete material: {barite_concrete}")
        
        # Magnetite concrete (high-density)
        magnetite_concrete = openmc.Material(name='Magnetite Concrete')
        magnetite_concrete.set_density('g/cm3', 3.53)
        magnetite_concrete.add_element('H', 0.001, 'wo')
        magnetite_concrete.add_element('O', 0.246, 'wo')
        magnetite_concrete.add_element('Na', 0.007, 'wo')
        magnetite_concrete.add_element('Mg', 0.006, 'wo')
        magnetite_concrete.add_element('Al', 0.021, 'wo')
        magnetite_concrete.add_element('Si', 0.084, 'wo')
        magnetite_concrete.add_element('P', 0.002, 'wo')
        magnetite_concrete.add_element('K', 0.008, 'wo')
        magnetite_concrete.add_element('Ca', 0.039, 'wo')
        magnetite_concrete.add_element('Ti', 0.016, 'wo')
        magnetite_concrete.add_element('Mn', 0.002, 'wo')
        magnetite_concrete.add_element('Fe', 0.568, 'wo')
        materials.append(magnetite_concrete)
        logger.info(f"Created magnetite concrete material: {magnetite_concrete}")
        
        # Air
        air = openmc.Material(name='Air')
        air.set_density('g/cm3', 0.001205)
        air.add_element('N', 0.7553, 'wo')
        air.add_element('O', 0.2318, 'wo')
        air.add_element('Ar', 0.0128, 'wo')
        air.add_element('C', 0.0001, 'wo')
        materials.append(air)
        logger.info(f"Created air material: {air}")
        
        # Tissue (ICRU sphere)
        tissue = openmc.Material(name='Tissue')
        tissue.set_density('g/cm3', 1.0)
        tissue.add_element('H', 0.1015, 'wo')
        tissue.add_element('C', 0.1124, 'wo')
        tissue.add_element('N', 0.0256, 'wo')
        tissue.add_element('O', 0.7621, 'wo')
        tissue.add_element('Na', 0.0013, 'wo')
        tissue.add_element('P', 0.0024, 'wo')
        tissue.add_element('S', 0.0022, 'wo')
        tissue.add_element('Cl', 0.0014, 'wo')
        tissue.add_element('K', 0.0021, 'wo')
        materials.append(tissue)
        logger.info(f"Created tissue material: {tissue}")
        
        # Export to XML for visualization and verification
        materials.export_to_xml()
        
        return materials, concrete, barite_concrete, magnetite_concrete, air, tissue
