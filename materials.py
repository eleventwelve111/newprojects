import openmc
import numpy as np

def create_materials():
    """Create materials for the simulation based on standard compositions"""
    materials = openmc.Materials()
    
    # ANSI/ANS-6.4-2006 Concrete composition
    concrete = openmc.Material(name='Concrete')
    # Ordinary concrete composition (ANSI/ANS-6.4-2006)
    concrete.add_element('H', 0.01, 'wo')
    concrete.add_element('C', 0.001, 'wo')
    concrete.add_element('O', 0.529107, 'wo')
    concrete.add_element('Na', 0.016, 'wo')
    concrete.add_element('Mg', 0.002, 'wo')
    concrete.add_element('Al', 0.033872, 'wo')
    concrete.add_element('Si', 0.337021, 'wo')
    concrete.add_element('K', 0.013, 'wo')
    concrete.add_element('Ca', 0.044, 'wo')
    concrete.add_element('Fe', 0.014, 'wo')
    concrete.set_density('g/cm3', 2.3)
    
    # Enhanced concrete type 1: Barite concrete
    barite_concrete = openmc.Material(name='Barite Concrete')
    barite_concrete.add_element('H', 0.0055, 'wo')
    barite_concrete.add_element('O', 0.311622, 'wo')
    barite_concrete.add_element('Mg', 0.001, 'wo')
    barite_concrete.add_element('Al', 0.004777, 'wo')
    barite_concrete.add_element('Si', 0.010019, 'wo')
    barite_concrete.add_element('S', 0.107759, 'wo')
    barite_concrete.add_element('Ca', 0.050045, 'wo')
    barite_concrete.add_element('Fe', 0.047557, 'wo')
    barite_concrete.add_element('Ba', 0.462722, 'wo')
    barite_concrete.set_density('g/cm3', 3.35)
    
    # Enhanced concrete type 2: Magnetite concrete (heavy)
    magnetite_concrete = openmc.Material(name='Magnetite Concrete')
    magnetite_concrete.add_element('H', 0.0055, 'wo')
    magnetite_concrete.add_element('O', 0.3164, 'wo')
    magnetite_concrete.add_element('Na', 0.004, 'wo')
    magnetite_concrete.add_element('Mg', 0.006, 'wo')
    magnetite_concrete.add_element('Al', 0.008, 'wo')
    magnetite_concrete.add_element('Si', 0.0935, 'wo')
    magnetite_concrete.add_element('Ca', 0.082, 'wo')
    magnetite_concrete.add_element('Fe', 0.4846, 'wo')
    magnetite_concrete.set_density('g/cm3', 3.9)
    
    # Air (standard dry air near sea level)
    air = openmc.Material(name='Air')
    air.add_element('N', 0.7553, 'wo')
    air.add_element('O', 0.2318, 'wo')
    air.add_element('Ar', 0.0128, 'wo')
    air.add_element('C', 0.0001, 'wo')
    air.set_density('g/cm3', 0.001225)
    
    # ICRU Tissue (for phantom)
    icru_tissue = openmc.Material(name='ICRU Tissue')
    icru_tissue.add_element('H', 0.1011, 'wo')
    icru_tissue.add_element('C', 0.1107, 'wo')
    icru_tissue.add_element('N', 0.0262, 'wo')
    icru_tissue.add_element('O', 0.7620, 'wo')
    icru_tissue.set_density('g/cm3', 1.0)
    
    materials.extend([concrete, barite_concrete, magnetite_concrete, air, icru_tissue])
    
    return materials, concrete, barite_concrete, magnetite_concrete, air, icru_tissue
