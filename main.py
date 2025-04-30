#!/usr/bin/env python3
"""
Main entry point for gamma-ray streaming simulation and analysis.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import openmc

from logging_utils import setup_logger, logger, LogSection
from simulation import setup_simulation, run_simulation, extract_results, save_results, run_parameter_study
from config import (WALL_THICKNESS, SOURCE_TO_WALL_DISTANCE, CHANNEL_DIAMETERS,
                   SOURCE_ENERGIES, DETECTOR_DISTANCES, DETECTOR_ANGLES, RESULTS_DIR)
from visualization import (create_dose_heatmap, plot_dose_vs_angle, plot_energy_spectrum,
                          plot_comparison_dose_calc_vs_simulation, plot_radiation_pattern)
from spectrum_analysis import analyze_energy_spectra
from streaming_analysis import analyze_streaming_effects
from compare_with_literature import compare_with_literature
from ml_analysis import perform_ml_analysis
from analysis_report import generate_report

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Gamma-ray streaming simulation through concrete shield")
    
    parser.add_argument("--mode", type=str, choices=["single", "study", "analyze", "report", "all"], 
                        default="all", help="Simulation mode")
    
    parser.add_argument("--energy", type=float, default=1.0,
                        help="Source energy in MeV (for single run)")
    
    parser.add_argument("--channel-diameter", type=float, default=0.5,
                        help="Channel diameter in cm (for single run)")
    
    parser.add_argument("--detector-distance", type=float, default=30.0,
                        help="Detector distance in cm (for single run)")
    
    parser.add_argument("--detector-angle", type=float, default=0.0,
                        help="Detector angle in degrees (for single run)")
    
    parser.add_argument("--particles", type=int, default=1000000,
                        help="Number of particles to simulate")
    
    parser.add_argument("--concrete-type", type=str, 
                        choices=["standard", "barite", "magnetite", "all"],
                        default="standard", help="Type of concrete to use")
    
    parser.add_argument("--use-weight-windows", action="store_true",
                        help="Use weight windows for variance reduction")
    
    parser.add_argument("--log-level", type=str, 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="INFO", help="Logging level")
    
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    
    return parser.parse_args()

def run_single_simulation(args):
    """Run a single simulation with specified parameters."""
    with LogSection("Single simulation run"):
        # Setup and run simulation
        model = setup_simulation(args.energy, args.channel_diameter, 
                               args.detector_distance, args.detector_angle,
                               args.concrete_type, args.particles)
        
        sp_filename = run_simulation(model, args.use_weight_windows)
        
        # Extract and save results
        results = extract_results(sp_filename, args.energy, args.channel_diameter, 
                                args.detector_distance, args.detector_angle)
        results['concrete_type'] = args.concrete_type
        
        output_path = save_results(results, args.output_dir)
        
        # Generate basic visualizations for this run
        create_dose_heatmap(results)
        plot_energy_spectrum(results)
        plot_radiation_pattern(results)
        
        logger.info(f"Single simulation completed. Results saved to {output_path}")
        
        return results

def run_full_study(args):
    """Run full parameter study."""
    with LogSection("Full parameter study"):
        if args.concrete_type == "all":
            concrete_types = ["standard", "barite", "magnetite"]
        else:
            concrete_types = [args.concrete_type]
        
        all_results = run_parameter_study(
            SOURCE_ENERGIES, CHANNEL_DIAMETERS, DETECTOR_DISTANCES, DETECTOR_ANGLES,
            concrete_types, args.particles, args.use_weight_windows
        )
        
        logger.info("Full parameter study completed")
        return all_results

def analyze_results(output_dir=None):
    """Analyze all available results."""
    with LogSection("Analyzing results"):
        if output_dir is None:
            output_dir = RESULTS_DIR / "data"
        else:
            output_dir = Path(output_dir)
        
        # Load combined results if available
        combined_file = output_dir / "combined_results.json"
        if combined_file.exists():
            import json
            with open(combined_file, 'r') as f:
                all_results = json.load(f)
        else:
            # Load individual result files
            result_files = list(output_dir.glob("results_*.json"))
            if not result_files:
                logger.error(f"No result files found in {output_dir}")
                return
            
            all_results = []
            for file in result_files:
                with open(file, 'r') as f:
                    all_results.append(json.load(f))
        
        logger.info(f"Loaded {len(all_results)} result sets for analysis")
        
        # Perform various analyses
        analyze_energy_spectra(all_results)
        analyze_streaming_effects(all_results)
        compare_with_literature(all_results)
        perform_ml_analysis(all_results)
        
        # Generate visualizations
        create_dose_heatmap(all_results)
        plot_dose_vs_angle(all_results)
        plot_comparison_dose_calc_vs_simulation(all_results)
        plot_radiation_pattern(all_results)
        
        logger.info("Analysis completed")
        return all_results

def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logger
    logger_name = "gamma_streaming"
    log_file = RESULTS_DIR / f"{logger_name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger(logger_name, log_file, getattr(sys.modules['logging'], args.log_level))
    
    logger.info("Starting gamma-ray streaming simulation and analysis")
    
    if args.mode == "single":
        results = run_single_simulation(args)
    
    elif args.mode == "study":
        results = run_full_study(args)
    
    elif args.mode == "analyze":
        results = analyze_results(args.output_dir)
    
    elif args.mode == "report":
        results = analyze_results(args.output_dir)
        generate_report(results, args.output_dir)
    
    elif args.mode == "all":
        # Run full study
        results = run_full_study(args)
        
        # Analyze results
        analyze_results(args.output_dir)
        
        # Generate report
        generate_report(results, args.output_dir)
    
    logger.info("Process completed successfully")

if __name__ == "__main__":
    main()
