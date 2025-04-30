#!/usr/bin/env python3
"""
Main simulation module for gamma-ray streaming through concrete shield.
"""
import os
import time
import json
import numpy as np
import openmc
from pathlib import Path
import matplotlib.pyplot as plt

from materials import create_materials
from source import create_point_source
from geometry import create_geometry
from tally import create_tallies
from dose import flux_to_dose_conversion
from logging_utils import logger, LogSection, timeit
from weight_windows import generate_weight_windows
from config import (WALL_THICKNESS, SOURCE_TO_WALL_DISTANCE, CHANNEL_DIAMETERS,
                   SOURCE_ENERGIES, DETECTOR_DISTANCES, DETECTOR_ANGLES, RESULTS_DIR)

@timeit
def setup_simulation(energy, channel_diameter, detector_distance, detector_angle, 
                    concrete_type="standard", particles=1000000, run_mode="fixed source"):
    """
    Set up an OpenMC simulation for gamma-ray streaming.
    
    Parameters:
    -----------
    energy : float
        Source energy in MeV
    channel_diameter : float
        Diameter of the air channel in cm
    detector_distance : float
        Distance of the detector from the wall in cm
    detector_angle : float
        Angle of the detector from the central axis in degrees
    concrete_type : str
        Type of concrete to use (standard, barite, or magnetite)
    particles : int
        Number of particles to simulate
    run_mode : str
        Run mode for the simulation
    
    Returns:
    --------
    model : openmc.Model
        OpenMC model
    """
    with LogSection(f"Setting up simulation (E={energy} MeV, Ø={channel_diameter} cm, "
                   f"dist={detector_distance} cm, angle={detector_angle}°)"):
        
        # Create materials
        materials_collection, concrete, barite_concrete, magnetite_concrete, air, tissue = create_materials()
        
        # Select concrete type
        if concrete_type == "barite":
            shield_material = barite_concrete
        elif concrete_type == "magnetite":
            shield_material = magnetite_concrete
        else:
            shield_material = concrete
        
        # Create geometry
        geometry = create_geometry(shield_material, air, tissue, channel_diameter, 
                                  detector_distance, detector_angle, concrete_type)
        
        # Create source
        source = create_point_source(energy, channel_diameter/2, biased=True)
        
        # Create tallies
        tallies = create_tallies(channel_diameter)
        
        # Create settings
        settings = openmc.Settings()
        settings.run_mode = run_mode
        settings.particles = particles
        settings.batches = 100
        settings.photon_transport = True
        settings.electron_treatment = 'ttb'  # Thick-target bremsstrahlung
        settings.source = source
        
        # Create the model
        model = openmc.Model(geometry=geometry, materials=materials_collection, 
                           settings=settings, tallies=tallies)
        
        return model

@timeit
def run_simulation(model, use_weight_windows=True, checkpoint_interval=10):
    """
    Run the OpenMC simulation with optional weight windows.
    
    Parameters:
    -----------
    model : openmc.Model
        OpenMC model
    use_weight_windows : bool
        Whether to use weight windows
    checkpoint_interval : int
        Number of batches between checkpoints
    
    Returns:
    --------
    sp_filename : str
        Statepoint filename
    """
    with LogSection("Running simulation"):
        if use_weight_windows:
            model = generate_weight_windows(model)
        
        # Set up checkpointing
        model.settings.checkpoint_batches = list(range(checkpoint_interval, 
                                               model.settings.batches, 
                                               checkpoint_interval))
        
        try:
            # Check if we can resume from a checkpoint
            checkpoint_files = sorted(Path('.').glob('statepoint.*.h5'))
            if checkpoint_files:
                latest_checkpoint = str(checkpoint_files[-1])
                logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
                sp_filename = model.run(restart_file=latest_checkpoint)
            else:
                sp_filename = model.run()
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise
            
        return sp_filename

@timeit
def extract_results(sp_filename, energy, channel_diameter, detector_distance, detector_angle):
    """
    Extract results from the statepoint file.
    
    Parameters:
    -----------
    sp_filename : str
        Statepoint filename
    energy : float
        Source energy in MeV
    channel_diameter : float
        Diameter of the air channel in cm
    detector_distance : float
        Distance of the detector from the wall in cm
    detector_angle : float
        Angle of the detector from the central axis in degrees
    
    Returns:
    --------
    results : dict
        Dictionary of results
    """
    with LogSection(f"Extracting results from {sp_filename}"):
        results = {
            'energy': energy,
            'channel_diameter': channel_diameter,
            'detector_distance': detector_distance,
            'detector_angle': detector_angle,
            'timestamp': time.time(),
            'dose': {},
            'flux': {},
            'kerma': {},
            'heating': {},
        }
        
        with openmc.StatePoint(sp_filename) as sp:
            # Extract dose tally
            dose_tally = sp.get_tally(name='dose_tally')
            results['dose']['value'] = float(dose_tally.mean)
            results['dose']['std_dev'] = float(dose_tally.std_dev)
            results['dose']['rel_error'] = float(dose_tally.std_dev / dose_tally.mean if dose_tally.mean > 0 else np.nan)
            
            # Extract flux tally
            flux_tally = sp.get_tally(name='flux_tally')
            results['flux']['value'] = float(flux_tally.mean)
            results['flux']['std_dev'] = float(flux_tally.std_dev)
            results['flux']['rel_error'] = float(flux_tally.std_dev / flux_tally.mean if flux_tally.mean > 0 else np.nan)
            
            # Extract kerma tally
            kerma_tally = sp.get_tally(name='kerma_tally')
            results['kerma']['value'] = float(kerma_tally.mean)
            results['kerma']['std_dev'] = float(kerma_tally.std_dev)
            results['kerma']['rel_error'] = float(kerma_tally.std_dev / kerma_tally.mean if kerma_tally.mean > 0 else np.nan)
            
            # Extract heating tally
            heating_tally = sp.get_tally(name='heating_tally')
            results['heating']['value'] = float(heating_tally.mean)
            results['heating']['std_dev'] = float(heating_tally.std_dev)
            results['heating']['rel_error'] = float(heating_tally.std_dev / heating_tally.mean if heating_tally.mean > 0 else np.nan)
            
            # Extract energy spectra if available
            try:
                spectra_tally = sp.get_tally(name='spectra_tally')
                energy_bins = spectra_tally.filters[0].bins
                energy_midpoints = 0.5 * (energy_bins[1:] + energy_bins[:-1])
                flux_spectrum = spectra_tally.get_values(scores=['flux']).flatten()
                
                results['energy_spectrum'] = {
                    'energy_bins': energy_bins.tolist(),
                    'energy_midpoints': energy_midpoints.tolist(),
                    'flux_spectrum': flux_spectrum.tolist()
                }
            except:
                logger.warning("Energy spectrum tally not found or could not be processed")
            
            # Extract mesh tally if available
            try:
                mesh_tally = sp.get_tally(name='mesh_tally')
                mesh_filter = mesh_tally.find_filter(openmc.MeshFilter)
                mesh = mesh_filter.mesh
                
                # Get dimensions
                nx, ny, nz = mesh.dimension
                
                # Extract values
                mesh_values = mesh_tally.get_values(scores=['flux']).reshape((nx, ny, nz))
                
                results['mesh_tally'] = {
                    'shape': mesh_values.shape,
                    'values': mesh_values.tolist()
                }
            except:
                logger.warning("Mesh tally not found or could not be processed")
        
        return results

@timeit
def save_results(results, output_dir=None):
    """
    Save simulation results to a JSON file.
    
    Parameters:
    -----------
    results : dict
        Dictionary of results
    output_dir : str or Path
        Directory to save results in
    
    Returns:
    --------
    filepath : Path
        Path to the saved results file
    """
    if output_dir is None:
        output_dir = RESULTS_DIR / "data"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    filename = (f"results_E{results['energy']}_"
               f"D{results['channel_diameter']}_"
               f"dist{results['detector_distance']}_"
               f"ang{results['detector_angle']}.json")
    
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {filepath}")
    return filepath

def run_parameter_study(energies=SOURCE_ENERGIES, 
                       channel_diameters=CHANNEL_DIAMETERS,
                       detector_distances=DETECTOR_DISTANCES,
                       detector_angles=DETECTOR_ANGLES,
                       concrete_types=["standard"],
                       particles=1000000,
                       use_weight_windows=True):
    """
    Run a parameter study varying source energy, channel diameter, detector distance and angle.
    
    Parameters:
    -----------
    energies : array-like
        Source energies in MeV
    channel_diameters : array-like
        Channel diameters in cm
    detector_distances : array-like
        Detector distances in cm
    detector_angles : array-like
        Detector angles in degrees
    concrete_types : list
        List of concrete types to use
    particles : int
        Number of particles per simulation
    use_weight_windows : bool
        Whether to use weight windows
    
    Returns:
    --------
    all_results : list
        List of result dictionaries
    """
    all_results = []
    
    total_runs = (len(energies) * len(channel_diameters) * 
                 len(detector_distances) * len(detector_angles) * len(concrete_types))
    
    logger.info(f"Starting parameter study with {total_runs} simulation runs")
    
    run_counter = 0
    
    for concrete_type in concrete_types:
        for energy in energies:
            for channel_diameter in channel_diameters:
                for detector_distance in detector_distances:
                    for detector_angle in detector_angles:
                        run_counter += 1
                        logger.info(f"Run {run_counter}/{total_runs}: "
                                   f"E={energy} MeV, Ø={channel_diameter} cm, "
                                   f"dist={detector_distance} cm, angle={detector_angle}°, "
                                   f"concrete={concrete_type}")
                        
                        # Setup and run simulation
                        model = setup_simulation(energy, channel_diameter, detector_distance, 
                                               detector_angle, concrete_type, particles)
                        
                        sp_filename = run_simulation(model, use_weight_windows)
                        
                        # Extract and save results
                        results = extract_results(sp_filename, energy, channel_diameter, 
                                                detector_distance, detector_angle)
                        results['concrete_type'] = concrete_type
                        
                        save_results(results)
                        all_results.append(results)
    
    # Save combined results
    output_dir = RESULTS_DIR / "data"
    combined_file = output_dir / "combined_results.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Parameter study completed. Combined results saved to {combined_file}")
    return all_results
