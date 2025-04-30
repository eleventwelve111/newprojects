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
    - Barite concrete (high-density, better for gamma shielding)
    - Magnetite concrete (high-density, improved attenuation)
    - Air
    - Tissue (ICRU sphere)
    - Void
    
    Returns:
        materials: OpenMC Materials collection
        materials_dict: Dictionary with all material objects
    """
    with LogSection("Creating materials"):
        # Create a material collection
        materials = openmc.Materials()
        materials_dict = {}
        
        # Standard concrete (ANSI/ANS-6.4-2006)
        concrete = openmc.Material(name='Standard Concrete')
        concrete.set_density('g/cm3', 2.3)
        concrete.add_element('H', 0.01, 'wo')
        concrete.add_element('C', 0.001, 'wo')
        concrete.add_element('O', 0.529, 'wo')
        concrete.add_element('Na', 0.016, 'wo')
        concrete.add_element('Mg', 0.002, 'wo')
        concrete.add_element('Al', 0.034, 'wo')
        concrete.add_element('Si', 0.337, 'wo')
        concrete.add_element('P', 0.002, 'wo')
        concrete.add_element('S', 0.002, 'wo')
        concrete.add_element('K', 0.013, 'wo')
        concrete.add_element('Ca', 0.044, 'wo')
        concrete.add_element('Fe', 0.014, 'wo')
        materials.append(concrete)
        materials_dict['standard_concrete'] = concrete
        
        # Barite concrete (ANSI/ANS-6.4-2006)
        barite_concrete = openmc.Material(name='Barite Concrete')
        barite_concrete.set_density('g/cm3', 3.35)
        barite_concrete.add_element('H', 0.0033, 'wo')
        barite_concrete.add_element('O', 0.311, 'wo')
        barite_concrete.add_element('Mg', 0.001, 'wo')
        barite_concrete.add_element('Al', 0.004, 'wo')
        barite_concrete.add_element('Si', 0.010, 'wo')
        barite_concrete.add_element('S', 0.107, 'wo')
        barite_concrete.add_element('Ca', 0.050, 'wo')
        barite_concrete.add_element('Fe', 0.047, 'wo')
        barite_concrete.add_element('Ba', 0.467, 'wo')
        materials.append(barite_concrete)
        materials_dict['barite_concrete'] = barite_concrete
        
        # Magnetite concrete (ANSI/ANS-6.4-2006)
        magnetite_concrete = openmc.Material(name='Magnetite Concrete')
        magnetite_concrete.set_density('g/cm3', 3.53)
        magnetite_concrete.add_element('H', 0.0055, 'wo')
        magnetite_concrete.add_element('O', 0.320, 'wo')
        magnetite_concrete.add_element('Mg', 0.006, 'wo')
        magnetite_concrete.add_element('Al', 0.008, 'wo')
        magnetite_concrete.add_element('Si', 0.035, 'wo')
        magnetite_concrete.add_element('P', 0.0007, 'wo')
        magnetite_concrete.add_element('S', 0.001, 'wo')
        magnetite_concrete.add_element('Ca', 0.060, 'wo')
        magnetite_concrete.add_element('Ti', 0.017, 'wo')
        magnetite_concrete.add_element('Mn', 0.002, 'wo')
        magnetite_concrete.add_element('Fe', 0.545, 'wo')
        materials.append(magnetite_concrete)
        materials_dict['magnetite_concrete'] = magnetite_concrete
        
        # Air (dry, near sea level)
        air = openmc.Material(name='Air')
        air.set_density('g/cm3', 0.001205)
        air.add_element('N', 0.75519, 'wo')
        air.add_element('O', 0.23179, 'wo')
        air.add_element('Ar', 0.01296, 'wo')
        air.add_element('C', 0.00006, 'wo')
        materials.append(air)
        materials_dict['air'] = air
        
        # ICRU tissue (ICRU sphere)
        tissue = openmc.Material(name='ICRU Tissue')
        tissue.set_density('g/cm3', 1.0)
        tissue.add_element('H', 0.101, 'wo')
        tissue.add_element('C', 0.111, 'wo')
        tissue.add_element('N', 0.026, 'wo')
        tissue.add_element('O', 0.762, 'wo')
        materials.append(tissue)
        materials_dict['tissue'] = tissue
        
        # Void (for external environment)
        void = openmc.Material(name='Void')
        void.set_density('g/cm3', 1e-10)
        void.add_element('H', 1.0)
        void.add_s_alpha_beta('c_H_in_H2O')
        materials.append(void)
        materials_dict['void'] = void
        
        return materials, materials_dict
