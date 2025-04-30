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
import pickle

from materials import create_materials
from source import create_point_source
from geometry import create_geometry
from tally import create_tallies
from dose import flux_to_dose_conversion
from logging_utils import logger, LogSection, timeit
from weight_windows import generate_weight_windows
from config import (WALL_THICKNESS, SOURCE_TO_WALL_DISTANCE, CHANNEL_DIAMETERS,
                   SOURCE_ENERGIES, DETECTOR_DISTANCES, DETECTOR_ANGLES, RESULTS_DIR,
                   PARTICLES_PER_BATCH, NUM_BATCHES, RANDOM_SEED, CHECKPOINT_FREQUENCY)

@timeit
def setup_simulation(energy, channel_diameter, detector_distance, detector_angle, 
                    concrete_type="standard", particles=None, run_mode="fixed source", use_weight_windows=True):
    """
    Set up an OpenMC simulation for gamma-ray streaming.
    
    Parameters:
    -----------
    energy : float
        Source energy in MeV
    channel_diameter : float
        Diameter of the air channel in cm
    detector_distance : float
        Distance of the detector from the back face of the wall in cm
    detector_angle : float
        Angle of the detector from the central axis in degrees
    concrete_type : str
        Type of concrete to use ("standard", "barite", or "magnetite")
    particles : int, optional
        Number of particles per batch (default from config)
    run_mode : str, optional
        Run mode ("fixed source" or "eigenvalue")
    use_weight_windows : bool, optional
        Whether to use weight windows for variance reduction
    
    Returns:
    --------
    model : dict
        Dictionary containing model components
    """
    with LogSection(f"Setting up simulation (E={energy} MeV, Ø={channel_diameter} cm, dist={detector_distance} cm, angle={detector_angle}°)"):
        # Create materials
        materials, materials_dict = create_materials()
        
        # Create geometry
        geometry, cells = create_geometry(materials_dict, channel_diameter, detector_distance, detector_angle, concrete_type)
        
        # Create source
        source = create_point_source(energy, channel_diameter/2, biased=True)
        
        # Create tallies
        tallies = create_tallies(cells)
        
        # Settings
        settings = openmc.Settings()
        settings.run_mode = run_mode
        
        if particles is None:
            particles = PARTICLES_PER_BATCH
            
        settings.particles = particles
        settings.batches = NUM_BATCHES
        settings.seed = RANDOM_SEED
        
        # Enable track output for visualization
        settings.track = (1, 100)  # Track first 100 particles
        
        # Create checkpoint file every N batches
        settings.checkpoint_batches = CHECKPOINT_FREQUENCY
        
        # Export to XML
        run_dir = Path(f"runs/E{energy}_D{channel_diameter}_dist{detector_distance}_ang{detector_angle}")
        run_dir.mkdir(exist_ok=True, parents=True)
        
        materials.export_to_xml(run_dir / "materials.xml")
        geometry.export_to_xml(run_dir / "geometry.xml")
        settings.export_to_xml(run_dir / "settings.xml")
        tallies.export_to_xml(run_dir / "tallies.xml")
        
        # Save the model components for later use
        model = {
            'materials': materials,
            'geometry': geometry,
            'source': source,
            'settings': settings,
            'tallies': tallies,
            'cells': cells,
            'run_dir': run_dir,
            'parameters': {
                'energy': energy,
                'channel_diameter': channel_diameter,
                'detector_distance': detector_distance,
                'detector_angle': detector_angle,
                'concrete_type': concrete_type
            }
        }
        
        # Save model to pickle file for later use
        with open(run_dir / "model.pkl", 'wb') as f:
            pickle.dump(model, f)
        
        # Generate weight windows if requested
        if use_weight_windows:
            weight_windows = generate_weight_windows(model, source, particles=int(particles*0.1))
            model['weight_windows'] = weight_windows
        
        logger.info(f"Simulation setup complete: {run_dir}")
        
        return model

@timeit
def run_simulation(model, resume=True):
    """
    Run an OpenMC simulation for gamma-ray streaming.
    
    Parameters:
    -----------
    model : dict
        Dictionary containing model components
    resume : bool, optional
        Whether to resume from a checkpoint file if available
    
    Returns:
    --------
    results : dict
        Dictionary of simulation results
    """
    with LogSection("Running simulation"):
        run_dir = model['run_dir']
                statepoint_file = run_dir / "statepoint.{0}.h5".format(model['settings'].batches)
        checkpoint_file = run_dir / "statepoint.restart.h5"
        
        # Check if checkpoint exists and we want to resume
        if resume and checkpoint_file.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_file}")
            openmc.resume_simulation()
        else:
            # Run the simulation from scratch
            logger.info("Starting new simulation")
            openmc.run(output=True)
        
        # Process results
        if statepoint_file.exists():
            # Load statepoint file
            sp = openmc.StatePoint(statepoint_file)
            
            # Extract results
            results = process_simulation_results(sp, model)
            
            # Save results to JSON
            results_file = run_dir / "results.json"
            with open(results_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = {}
                for key, value in results.items():
                    if isinstance(value, np.ndarray):
                        serializable_results[key] = value.tolist()
                    elif isinstance(value, dict):
                        serializable_results[key] = {}
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, np.ndarray):
                                serializable_results[key][sub_key] = sub_value.tolist()
                            else:
                                serializable_results[key][sub_key] = sub_value
                    else:
                        serializable_results[key] = value
                
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Results saved to {results_file}")
            
            return results
        else:
            logger.error(f"Statepoint file not found: {statepoint_file}")
            return None

@timeit
def process_simulation_results(sp, model):
    """
    Process simulation results.
    
    Parameters:
    -----------
    sp : openmc.StatePoint
        OpenMC StatePoint object
    model : dict
        Dictionary of model components
    
    Returns:
    --------
    results : dict
        Dictionary of simulation results
    """
    with LogSection("Processing simulation results"):
        # Get the parameters
        parameters = model['parameters']
        
        # Get results from tallies
        flux_tally = sp.get_tally(name='Detector Flux')
        heating_tally = sp.get_tally(name='Detector Heating')
        kerma_tally = sp.get_tally(name='Detector Kerma')
        dose_tally = sp.get_tally(name='Detector Dose')
        mesh_tally = sp.get_tally(name='Mesh Tally')
        fine_mesh_tally = sp.get_tally(name='Fine Mesh Tally')
        
        # Extract energy grid from flux tally
        energy_filter = flux_tally.find_filter(openmc.EnergyFilter)
        energy_bins = energy_filter.bins
        energy_midpoints = np.sqrt(energy_bins[1:] * energy_bins[:-1])
        
        # Extract flux spectrum
        flux_spectrum = flux_tally.get_values(scores=['flux'])
        flux_spectrum_error = flux_tally.std_dev
        
        # Calculate total flux
        total_flux = np.sum(flux_spectrum)
        
        # Calculate dose using flux-to-dose conversion factors
        dose_values = np.zeros_like(flux_spectrum)
        for i, energy in enumerate(energy_midpoints):
            dose_values[i] = flux_spectrum[i] * flux_to_dose_conversion(energy)
        
        total_dose = np.sum(dose_values)
        
        # Get heating and kerma values
        heating_value = heating_tally.get_values(scores=['heating']).flatten()[0]
        kerma_value = kerma_tally.get_values(scores=['kerma-photon']).flatten()[0]
        
        # Extract mesh tally data
        mesh_shape = mesh_tally.find_filter(openmc.MeshFilter).mesh.dimension
        mesh_values = mesh_tally.get_values(scores=['flux']).reshape(mesh_shape)
        
        # Extract fine mesh tally data
        fine_mesh_shape = fine_mesh_tally.find_filter(openmc.MeshFilter).mesh.dimension
        fine_mesh_values = fine_mesh_tally.get_values(scores=['flux']).reshape(fine_mesh_shape)
        
        # Assemble results
        results = {
            'parameters': parameters,
            'energy_bins': energy_bins.tolist(),
            'energy_midpoints': energy_midpoints.tolist(),
            'flux_spectrum': flux_spectrum.flatten().tolist(),
            'flux_spectrum_error': flux_spectrum_error.flatten().tolist(),
            'total_flux': float(total_flux),
            'dose_values': dose_values.flatten().tolist(),
            'total_dose': float(total_dose),
            'heating_value': float(heating_value),
            'kerma_value': float(kerma_value),
            'mesh_shape': mesh_shape,
            'fine_mesh_shape': fine_mesh_shape
        }
        
        logger.info(f"Total flux: {total_flux:.4e} particles/cm²/source_particle")
        logger.info(f"Total dose: {total_dose:.4e} rem/hr/source_intensity")
        logger.info(f"Heating: {heating_value:.4e} MeV/g/source_particle")
        logger.info(f"Kerma: {kerma_value:.4e} MeV/g/source_particle")
        
        return results

@timeit
def parameter_study():
    """
    Perform a parameter study over energy, channel diameter, detector distance, and angle.
    
    Returns:
    --------
    all_results : list
        List of result dictionaries
    """
    with LogSection("Starting parameter study"):
        all_results = []
        
        # Define progress counter
        total_runs = len(SOURCE_ENERGIES) * len(CHANNEL_DIAMETERS) * len(DETECTOR_DISTANCES) * len(DETECTOR_ANGLES)
        current_run = 0
        
        # Loop over all parameters
        for energy in SOURCE_ENERGIES:
            for diameter in CHANNEL_DIAMETERS:
                for distance in DETECTOR_DISTANCES:
                    for angle in DETECTOR_ANGLES:
                        current_run += 1
                        logger.info(f"Run {current_run}/{total_runs}: E={energy} MeV, D={diameter} cm, dist={distance} cm, angle={angle}°")
                        
                        # Setup simulation
                        model = setup_simulation(
                            energy=energy,
                            channel_diameter=diameter,
                            detector_distance=distance,
                            detector_angle=angle,
                            concrete_type="standard"
                        )
                        
                        # Run simulation
                        results = run_simulation(model)
                        
                        if results:
                            all_results.append(results)
                            
                            # Save progress after each run
                            results_file = RESULTS_DIR / "parameter_study_results.json"
                            with open(results_file, 'w') as f:
                                # Convert numpy arrays to lists for JSON serialization
                                serializable_results = []
                                for result in all_results:
                                    serializable_result = {}
                                    for key, value in result.items():
                                        if isinstance(value, np.ndarray):
                                            serializable_result[key] = value.tolist()
                                        elif isinstance(value, dict):
                                            serializable_result[key] = {}
                                            for sub_key, sub_value in value.items():
                                                if isinstance(sub_value, np.ndarray):
                                                    serializable_result[key][sub_key] = sub_value.tolist()
                                                else:
                                                    serializable_result[key][sub_key] = sub_value
                                        else:
                                            serializable_result[key] = value
                                    serializable_results.append(serializable_result)
                                
                                json.dump(serializable_results, f, indent=2)
                        
                        logger.info(f"Completed {current_run}/{total_runs} runs")
        
        logger.info(f"Parameter study complete: {len(all_results)} results")
        return all_results

@timeit
def concrete_comparison_study():
    """
    Compare different concrete types for shielding effectiveness.
    
    Returns:
    --------
    comparison_results : dict
        Dictionary of results for different concrete types
    """
    with LogSection("Starting concrete comparison study"):
        concrete_types = ["standard", "barite", "magnetite"]
        comparison_results = {}
        
        # Use fixed parameters for comparison
        energy = 1.0  # MeV
        diameter = 0.5  # cm
        distance = 30  # cm
        angle = 0  # degrees
        
        for concrete_type in concrete_types:
            logger.info(f"Running simulation with {concrete_type} concrete")
            
            # Setup simulation
            model = setup_simulation(
                energy=energy,
                channel_diameter=diameter,
                detector_distance=distance,
                detector_angle=angle,
                concrete_type=concrete_type
            )
            
            # Run simulation
            results = run_simulation(model)
            
            if results:
                comparison_results[concrete_type] = results
        
        # Save results
        results_file = RESULTS_DIR / "concrete_comparison_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for concrete_type, result in comparison_results.items():
                serializable_results[concrete_type] = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        serializable_results[concrete_type][key] = value.tolist()
                    elif isinstance(value, dict):
                        serializable_results[concrete_type][key] = {}
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, np.ndarray):
                                serializable_results[concrete_type][key][sub_key] = sub_value.tolist()
                            else:
                                serializable_results[concrete_type][key][sub_key] = sub_value
                    else:
                        serializable_results[concrete_type][key] = value
            
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Concrete comparison study complete")
        return comparison_results

