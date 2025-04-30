#!/usr/bin/env python3
"""
Machine learning analysis for gamma-ray streaming simulation through concrete shield.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE

from logging_utils import logger, LogSection, timeit
from config import RESULTS_DIR, PLOTS_DIR, DATA_DIR

@timeit
def prepare_ml_dataset(results):
    """
    Prepare dataset for machine learning from simulation results.
    
    Parameters:
    -----------
    results : list
        List of simulation result dictionaries
    
    Returns:
    --------
    dataset : pandas.DataFrame
        Dataset ready for machine learning
    """
    with LogSection("Preparing machine learning dataset"):
        # Extract features and target from results
        data = []
        
        for result in results:
            # Skip results without dose or with NaN/inf values
                        # Skip results without dose or with NaN/inf values
            if 'dose' not in result or not np.isfinite(result['dose']['value']):
                continue
            
            # Basic features
            features = {
                'energy': result['energy'],
                'channel_diameter': result['channel_diameter'],
                'detector_distance': result['detector_distance'],
                'detector_angle': result['detector_angle']
            }
            
            # Add derived features
            features['aspect_ratio'] = result.get('wall_thickness', WALL_THICKNESS) / features['channel_diameter'] if features['channel_diameter'] > 0 else 1000
            features['solid_angle'] = np.pi * (features['channel_diameter'] / 2)**2 / (features['detector_distance']**2) if features['detector_distance'] > 0 else 0
            features['l_d_ratio'] = result.get('wall_thickness', WALL_THICKNESS) / features['channel_diameter'] if features['channel_diameter'] > 0 else 1000
            
            # Target variable (dose)
            target = {
                'dose': result['dose']['value'],
                'dose_error': result['dose'].get('error', 0.0)
            }
            
            # Add uncertainty information (if available) as sample weight
            weight = 1.0 / (target['dose_error']**2) if target['dose_error'] > 0 else 1.0
            
            # Combine features and target
            entry = {**features, **target, 'weight': weight}
            data.append(entry)
        
        # Convert to DataFrame
        dataset = pd.DataFrame(data)
        
        logger.info(f"Created dataset with {len(dataset)} samples and {len(dataset.columns)} features")
        
        # Save dataset to CSV
        output_file = DATA_DIR / "ml_dataset.csv"
        dataset.to_csv(output_file, index=False)
        logger.info(f"Saved dataset to {output_file}")
        
        return dataset

@timeit
def feature_engineering(dataset):
    """
    Perform feature engineering on the dataset.
    
    Parameters:
    -----------
    dataset : pandas.DataFrame
        Input dataset
    
    Returns:
    --------
    enhanced_dataset : pandas.DataFrame
        Dataset with engineered features
    feature_importance : dict
        Dictionary of feature importance scores
    """
    with LogSection("Performing feature engineering"):
        # Create a copy of the dataset
        enhanced_dataset = dataset.copy()
        
        # Add polynomial features
        enhanced_dataset['energy_squared'] = enhanced_dataset['energy'] ** 2
        enhanced_dataset['energy_cubed'] = enhanced_dataset['energy'] ** 3
        enhanced_dataset['diameter_squared'] = enhanced_dataset['channel_diameter'] ** 2
        enhanced_dataset['distance_squared'] = enhanced_dataset['detector_distance'] ** 2
        enhanced_dataset['angle_squared'] = enhanced_dataset['detector_angle'] ** 2
        
        # Add logarithmic features
        enhanced_dataset['log_energy'] = np.log1p(enhanced_dataset['energy'])
        enhanced_dataset['log_diameter'] = np.log1p(enhanced_dataset['channel_diameter'])
        enhanced_dataset['log_distance'] = np.log1p(enhanced_dataset['detector_distance'])
        enhanced_dataset['log_aspect_ratio'] = np.log1p(enhanced_dataset['aspect_ratio'])
        
        # Add interaction terms
        enhanced_dataset['energy_x_diameter'] = enhanced_dataset['energy'] * enhanced_dataset['channel_diameter']
        enhanced_dataset['energy_x_distance'] = enhanced_dataset['energy'] * enhanced_dataset['detector_distance']
        enhanced_dataset['energy_x_angle'] = enhanced_dataset['energy'] * enhanced_dataset['detector_angle']
        enhanced_dataset['diameter_x_distance'] = enhanced_dataset['channel_diameter'] * enhanced_dataset['detector_distance']
        
        # Add physics-based features
        # Attenuation-like feature
        enhanced_dataset['atten_like'] = np.exp(-0.1 * enhanced_dataset['energy']) * \
                                     (enhanced_dataset['channel_diameter'] + 1) / \
                                     (enhanced_dataset['detector_distance'] ** 2)
        
        # Estimate direct beam component
        enhanced_dataset['direct_component'] = (enhanced_dataset['channel_diameter'] ** 2) / \
                                          (4 * enhanced_dataset['detector_distance'] ** 2 + enhanced_dataset['detector_angle'] ** 2)
        
        # Solid angle correction for off-axis points
        enhanced_dataset['solid_angle_corrected'] = enhanced_dataset['solid_angle'] * \
                                             np.cos(np.radians(enhanced_dataset['detector_angle']))
        
        # Scattering component estimation
        enhanced_dataset['scatter_component'] = enhanced_dataset['energy'] * enhanced_dataset['channel_diameter'] / \
                                          (enhanced_dataset['detector_distance'] * (1 + enhanced_dataset['detector_angle']))
        
        # Create log of dose (for log-linear models)
        enhanced_dataset['log_dose'] = np.log(enhanced_dataset['dose'])
        
        # Assess feature importance using a Random Forest
        feature_cols = [col for col in enhanced_dataset.columns 
                      if col not in ['dose', 'dose_error', 'weight', 'log_dose']]
        
        X = enhanced_dataset[feature_cols]
        y = enhanced_dataset['dose']
        weights = enhanced_dataset['weight']
        
        # Normalize weights
        weights = weights / weights.mean()
        
        # Train a Random Forest to get feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y, sample_weight=weights)
        
        # Store feature importance
        feature_importance = {
            'features': feature_cols,
            'importance': rf.feature_importances_
        }
        
        # Sort features by importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 features by importance:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Save enhanced dataset
        output_file = DATA_DIR / "ml_dataset_enhanced.csv"
        enhanced_dataset.to_csv(output_file, index=False)
        logger.info(f"Saved enhanced dataset to {output_file}")
        
        return enhanced_dataset, feature_importance

@timeit
def train_ml_models(dataset, target_col='dose', test_size=0.2):
    """
    Train and evaluate multiple machine learning models.
    
    Parameters:
    -----------
    dataset : pandas.DataFrame
        Input dataset with features and target
    target_col : str
        Column name for the target variable
    test_size : float
        Fraction of data to use for testing
    
    Returns:
    --------
    model_results : dict
        Dictionary of model results and performance metrics
    """
    with LogSection("Training machine learning models"):
        # Split features and target
        feature_cols = [col for col in dataset.columns 
                      if col not in ['dose', 'dose_error', 'log_dose', 'weight']]
        
        X = dataset[feature_cols]
        y = dataset[target_col]
        weights = dataset['weight']
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, weights, test_size=test_size, random_state=42
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        
        # Dictionary to store models and results
        models = {}
        
        # 1. Linear Regression with polynomial features
        poly_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('model', LinearRegression())
        ])
        poly_pipeline.fit(X_train, y_train, model__sample_weight=w_train)
        models['polynomial'] = {
            'name': 'Polynomial Regression',
            'model': poly_pipeline,
            'train_score': poly_pipeline.score(X_train, y_train, sample_weight=w_train),
            'test_score': poly_pipeline.score(X_test, y_test, sample_weight=w_test),
            'y_pred': poly_pipeline.predict(X_test)
        }
        
        # 2. Ridge Regression (L2 regularization)
        ridge_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('model', Ridge(alpha=1.0))
        ])
        ridge_pipeline.fit(X_train, y_train, model__sample_weight=w_train)
        models['ridge'] = {
            'name': 'Ridge Regression',
            'model': ridge_pipeline,
            'train_score': ridge_pipeline.score(X_train, y_train, sample_weight=w_train),
            'test_score': ridge_pipeline.score(X_test, y_test, sample_weight=w_test),
            'y_pred': ridge_pipeline.predict(X_test)
        }
        
        # 3. Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train, sample_weight=w_train)
        models['random_forest'] = {
            'name': 'Random Forest',
            'model': rf_model,
            'train_score': rf_model.score(X_train, y_train, sample_weight=w_train),
            'test_score': rf_model.score(X_test, y_test, sample_weight=w_test),
            'y_pred': rf_model.predict(X_test),
            'feature_importance': dict(zip(feature_cols, rf_model.feature_importances_))
        }
        
        # 4. Gradient Boosting Regressor
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train, sample_weight=w_train)
        models['gradient_boosting'] = {
            'name': 'Gradient Boosting',
            'model': gb_model,
            'train_score': gb_model.score(X_train, y_train, sample_weight=w_train),
            'test_score': gb_model.score(X_test, y_test, sample_weight=w_test),
            'y_pred': gb_model.predict(X_test),
            'feature_importance': dict(zip(feature_cols, gb_model.feature_importances_))
        }
        
        # 5. Support Vector Regression (with RBF kernel)
        svr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1))
        ])
        svr_pipeline.fit(X_train, y_train, model__sample_weight=w_train)
        models['svr'] = {
            'name': 'Support Vector Regression',
            'model': svr_pipeline,
            'train_score': svr_pipeline.score(X_train, y_train, sample_weight=w_train),
            'test_score': svr_pipeline.score(X_test, y_test, sample_weight=w_test),
            'y_pred': svr_pipeline.predict(X_test)
        }
        
        # Calculate additional metrics for each model
        for name, model_data in models.items():
            y_pred = model_data['y_pred']
            
            model_data['metrics'] = {
                'r2': r2_score(y_test, y_pred, sample_weight=w_test),
                'mse': mean_squared_error(y_test, y_pred, sample_weight=w_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred, sample_weight=w_test)),
                'mae': mean_absolute_error(y_test, y_pred, sample_weight=w_test)
            }
            
            logger.info(f"{model_data['name']} - R²: {model_data['metrics']['r2']:.4f}, "
                      f"RMSE: {model_data['metrics']['rmse']:.4f}")
        
        # Determine the best model based on test R² score
        best_model = max(models.items(), key=lambda x: x[1]['metrics']['r2'])
        logger.info(f"Best model: {best_model[1]['name']} (R² = {best_model[1]['metrics']['r2']:.4f})")
        
        # Save models
        models_dir = DATA_DIR / "ml_models"
        models_dir.mkdir(exist_ok=True, parents=True)
        
        for name, model_data in models.items():
            model_file = models_dir / f"{name}_model.joblib"
            joblib.dump(model_data['model'], model_file)
            logger.info(f"Saved {name} model to {model_file}")
        
        # Return all models and test data for visualization
        return {
            'models': models,
            'best_model': best_model[0],
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': feature_cols
        }

@timeit
def perform_model_optimization(dataset, target_col='dose'):
    """
    Perform hyperparameter optimization for the best models.
    
    Parameters:
    -----------
    dataset : pandas.DataFrame
        Input dataset with features and target
    target_col : str
        Column name for the target variable
    
    Returns:
    --------
    optimized_models : dict
        Dictionary of optimized models and their performance
    """
    with LogSection("Optimizing machine learning models"):
        # Split features and target
        feature_cols = [col for col in dataset.columns 
                      if col not in ['dose', 'dose_error', 'log_dose', 'weight']]
        
        X = dataset[feature_cols]
        y = dataset[target_col]
        weights = dataset['weight']
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X, y, weights, test_size=0.2, random_state=42
        )
        
        # Define models and parameter grids
        model_grids = {
            'ridge': {
                'model': Ridge(),
                'params': {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                }
            },
            'random_forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                                        'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
            }
        }
        
        optimized_models = {}
        
        # Perform grid search for each model
        for name, model_info in model_grids.items():
            logger.info(f"Optimizing {name} model...")
            
            # Create the pipeline
            if name == 'ridge':
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                    ('model', model_info['model'])
                ])
                
                # Adjust parameter grid for pipeline
                param_grid = {'model__' + k: v for k, v in model_info['params'].items()}
            else:
                pipeline = model_info['model']
                param_grid = model_info['params']
            
            # Define CV strategy with sample weights
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            
            # Perform grid search
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            
            # Fit the model
            if name == 'ridge':
                grid_search.fit(X_train, y_train, model__sample_weight=w_train)
            else:
                grid_search.fit(X_train, y_train, sample_weight=w_train)
            
            # Get the best model
            best_model = grid_search.best_estimator_
            
            # Evaluate on validation set
            val_score = best_model.score(X_val, y_val, sample_weight=w_val)
            val_pred = best_model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_pred, sample_weight=w_val)
            val_rmse = np.sqrt(val_mse)
            val_r2 = r2_score(y_val, val_pred, sample_weight=w_val)
            
            logger.info(f"Best parameters for {name}: {grid_search.best_params_}")
            logger.info(f"Validation R²: {val_r2:.4f}, RMSE: {val_rmse:.4f}")
            
            # Store the optimized model
            optimized_models[name] = {
                'name': name,
                'model': best_model,
                'best_params': grid_search.best_params_,
                'validation_score': val_score,
                'validation_r2': val_r2,
                'validation_rmse': val_rmse,
                'cross_val_results': grid_search.cv_results_
            }
            
            # Save the optimized model
            model_file = DATA_DIR / "ml_models" / f"{name}_optimized.joblib"
            joblib.dump(best_model, model_file)
            logger.info(f"Saved optimized {name} model to {model_file}")
        
        # Determine the best optimized model
        best_model_name = max(optimized_models.items(), key=lambda x: x[1]['validation_r2'])[0]
        logger.info(f"Best optimized model: {best_model_name} "
                  f"(R² = {optimized_models[best_model_name]['validation_r2']:.4f})")
        
        # Return the optimized models
        optimized_models['best_model'] = best_model_name
        
        return optimized_models

@timeit
def plot_ml_results(ml_results, output_dir=None):
    """
    Create plots for machine learning results.
    
    Parameters:
    -----------
    ml_results : dict
        Dictionary of machine learning results
    output_dir : Path or str, optional
        Directory to save plots
    
    Returns:
    --------
    figs : list
        List of figures created
    """
    with LogSection("Creating machine learning plots"):
        if output_dir is None:
            output_dir = PLOTS_DIR / "ml_analysis"
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        figs = []
        
        # 1. Actual vs. Predicted plot for each model
        if 'models' in ml_results:
            models = ml_results['models']
            X_test = ml_results['X_test']
            y_test = ml_results['y_test']
            
            # Create scatter plot for each model
            for name, model_data in models.items():
                fig, ax = plt.subplots(figsize=(10, 8))
                
                y_pred = model_data['y_pred']
                
                # Determine plot range
                min_val = min(min(y_test), min(y_pred))
                max_val = max(max(y_test), max(y_pred))
                buffer = 0.1 * (max_val - min_val)
                plot_min = max(0, min_val - buffer)
                plot_max = max_val + buffer
                
                # Create scatter plot
                ax.scatter(y_test, y_pred, alpha=0.6, s=30)
                
                # Add identity line
                ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', lw=2)
                
                # Add error bounds (±20%)
                ax.plot([plot_min, plot_max], [0.8*plot_min, 0.8*plot_max], 'r:', lw=1, alpha=0.7)
                ax.plot([plot_min, plot_max], [1.2*plot_min, 1.2*plot_max], 'r:', lw=1, alpha=0.7)
                
                # Add metrics to plot
                metrics = model_data['metrics']
                ax.text(0.05, 0.95, 
                       f"R² = {metrics['r2']:.4f}\nRMSE = {metrics['rmse']:.4f}\nMAE = {metrics['mae']:.4f}",
                       transform=ax.transAxes, fontsize=12, va='top', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_title(f'Actual vs. Predicted - {model_data["name"]}', fontsize=14)
                ax.set_xlabel('Actual Dose', fontsize=12)
                ax.set_ylabel('Predicted Dose', fontsize=12)
                
                # Use log scale if range is large
                if max_val / (min_val + 1e-10) > 100:
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Save figure
                filename = f"actual_vs_predicted_{name}.png"
                plt.tight_layout()
                plt.savefig(output_dir / filename, dpi=300)
                plt.close()
                logger.info(f"Created actual vs. predicted plot: {filename}")
                
                figs.append(fig)
            
            # 2. Model comparison plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Collect metrics for all models
            model_names = []
            r2_scores = []
            rmse_scores = []
            
            for name, model_data in models.items():
                model_names.append(model_data['name'])
                r2_scores.append(model_data['metrics']['r2'])
                rmse_scores.append(model_data['metrics']['rmse'])
            
            # Create bar chart
            x = np.arange(len(model_names))
            width = 0.35
            
            ax.bar(x - width/2, r2_scores, width, label='R² Score')
            
            # Add RMSE on secondary y-axis
            ax2 = ax.twinx()
            ax2.bar(x + width/2, rmse_scores, width, color='orange', label='RMSE')
            
            # Customize plot
            ax.set_title('Model Performance Comparison', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_ylabel('R² Score', fontsize=12)
            ax2.set_ylabel('RMSE', fontsize=12)
            
            # Add legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Save figure
            filename = "model_comparison.png"
            plt.tight_layout()
            plt.savefig(output_dir / filename, dpi=300)
            plt.close()
            logger.info(f"Created model comparison plot: {filename}")
            
            figs.append(fig)
            
            # 3. Feature importance plot (for tree-based models)
            for name in ['random_forest', 'gradient_boosting']:
                if name in models and 'feature_importance' in models[name]:
                    model_data = models[name]
                    feature_importance = model_data['feature_importance']
                    
                    # Sort features by importance
                    sorted_idx = np.argsort([feature_importance[f] for f in ml_results['feature_names']])
                    
                    fig, ax = plt.subplots(figsize=(10, 12))
                    
                    y_pos = np.arange(len(ml_results['feature_names']))
                    importance = [feature_importance[f] for f in np.array(ml_results['feature_names'])[sorted_idx]]
                    features = np.array(ml_results['feature_names'])[sorted_idx]
                    
                    ax.barh(y_pos, importance, align='center')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(features)
                    ax.invert_yaxis()  # labels read top-to-bottom
                    ax.set_title(f'Feature Importance - {model_data["name"]}', fontsize=14)
                    ax.set_xlabel('Importance', fontsize=12)
                    
                    # Save figure
                    filename = f"feature_importance_{name}.png"
                    plt.tight_layout()
                    plt.savefig(output_dir / filename, dpi=300)
                    plt.close()
                    logger.info(f"Created feature importance plot: {filename}")
                    
                    figs.append(fig)
            
            # 4. Residual plot for the best model
            best_model_name = ml_results['best_model']
            best_model_data = models[best_model_name]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            y_pred = best_model_data['y_pred']
            residuals = y_test - y_pred
            
            # Create scatter plot of residuals
            ax.scatter(y_pred, residuals, alpha=0.6, s=30)
            
            # Add horizontal line at y=0
            ax.axhline(y=0, color='k', linestyle='--', lw=2)
            
            # Add error bounds
            std_resid = np.std(residuals)
            ax.axhline(y=2*std_resid, color='r', linestyle=':', lw=1, alpha=0.7)
            ax.axhline(y=-2*std_resid, color='r', linestyle=':', lw=1, alpha=0.7)
            
            ax.set_title(f'Residual Plot - {best_model_data["name"]} (Best Model)', fontsize=14)
            ax.set_xlabel('Predicted Dose', fontsize=12)
            ax.set_ylabel('Residual (Actual - Predicted)', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Save figure
            filename = "residual_plot_best_model.png"
            plt.tight_layout()
            plt.savefig(output_dir / filename, dpi=300)
            plt.close()
            logger.info(f"Created residual plot: {filename}")
            
            figs.append(fig)
        
        # 5. Hyperparameter tuning results (if available)
        if 'optimized_models' in ml_results:
            optimized_models = ml_results['optimized_models']
            
            for name, model_data in optimized_models.items():
                if name == 'best_model':
                    continue
                
                if 'cross_val_results' in model_data:
                    # Extract relevant results for visualization
                    results = model_data['cross_val_results']
                    
                    # Find the hyperparameter with the most variation
                    param_names = [p for p in results['params'][0].keys() 
                                 if p.startswith('model__') or not p.startswith('model')]
                    
                    if not param_names:
                        continue
                    
                    # If there are multiple parameters, create a grid of plots
                    n_params = len(param_names)
                    
                    if n_params == 1:
                        # Simple line plot for one parameter
                        param = param_names[0]
                        param_values = sorted(list(set([p[param] for p in results['params']])))
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Map param values to their scores
                        param_scores = []
                        param_stds = []
                        
                        for value in param_values:
                            indices = [i for i, p in enumerate(results['params']) if p[param] == value]
                            scores = [-results['mean_test_score'][i] for i in indices]  # Convert back from negative
                            stds = [results['std_test_score'][i] for i in indices]
                            
                            param_scores.append(np.mean(scores))
                            param_stds.append(np.mean(stds))
                        
                        # Plot mean score with error bars
                        ax.errorbar(param_values, param_scores, yerr=param_stds, 
                                  marker='o', linestyle='-', capsize=5)
                        
                                                ax.set_title(f'Hyperparameter Tuning - {name.title()}', fontsize=14)
                        ax.set_xlabel(param.replace('model__', '').replace('_', ' ').title(), fontsize=12)
                        ax.set_ylabel('Mean Squared Error', fontsize=12)
                        ax.grid(True, linestyle='--', alpha=0.7)
                        
                        # Mark the best value
                        best_value = model_data['best_params'][param]
                        best_idx = param_values.index(best_value)
                        ax.plot(best_value, param_scores[best_idx], 'ro', markersize=10, 
                              label=f'Best: {best_value}')
                        ax.legend()
                        
                    else:
                        # Create a grid for multiple parameters
                        n_cols = min(n_params, 3)
                        n_rows = (n_params + n_cols - 1) // n_cols
                        
                        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
                        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
                        
                        for i, param in enumerate(param_names):
                            ax = axes[i]
                            param_values = sorted(list(set([p[param] for p in results['params']])))
                            
                            # Map param values to their scores
                            param_scores = []
                            param_stds = []
                            
                            for value in param_values:
                                indices = [i for i, p in enumerate(results['params']) if p[param] == value]
                                scores = [-results['mean_test_score'][i] for i in indices]
                                stds = [results['std_test_score'][i] for i in indices]
                                
                                param_scores.append(np.mean(scores))
                                param_stds.append(np.mean(stds))
                            
                            # Plot mean score with error bars
                            ax.errorbar(range(len(param_values)), param_scores, yerr=param_stds, 
                                      marker='o', linestyle='-', capsize=5)
                            
                            ax.set_title(param.replace('model__', '').replace('_', ' ').title(), fontsize=12)
                            ax.set_xticks(range(len(param_values)))
                            ax.set_xticklabels(param_values, rotation=45 if len(str(param_values[0])) > 5 else 0)
                            ax.set_ylabel('MSE', fontsize=10)
                            ax.grid(True, linestyle='--', alpha=0.7)
                            
                            # Mark the best value
                            best_value = model_data['best_params'][param]
                            best_idx = param_values.index(best_value)
                            ax.plot(best_idx, param_scores[best_idx], 'ro', markersize=8)
                        
                        # Hide unused subplots
                        for j in range(i+1, len(axes)):
                            axes[j].set_visible(False)
                    
                    # Save figure
                    filename = f"hyperparameter_tuning_{name}.png"
                    plt.tight_layout()
                    plt.savefig(output_dir / filename, dpi=300)
                    plt.close()
                    logger.info(f"Created hyperparameter tuning plot: {filename}")
                    
                    figs.append(fig)
        
        return figs

@timeit
def make_dose_predictions(ml_results, new_data=None):
    """
    Make dose predictions using the best model.
    
    Parameters:
    -----------
    ml_results : dict
        Dictionary of machine learning results
    new_data : pandas.DataFrame, optional
        New data to make predictions on
    
    Returns:
    --------
    predictions : dict
        Dictionary of predictions and analysis results
    """
    with LogSection("Making dose predictions"):
        # Get the best model
        if 'optimized_models' in ml_results and ml_results['optimized_models']['best_model'] != 'best_model':
            # Use optimized model if available
            best_model_name = ml_results['optimized_models']['best_model']
            best_model = ml_results['optimized_models'][best_model_name]['model']
            logger.info(f"Using optimized {best_model_name} model for predictions")
        else:
            # Use original model
            best_model_name = ml_results['best_model']
            best_model = ml_results['models'][best_model_name]['model']
            logger.info(f"Using {best_model_name} model for predictions")
        
        if new_data is None:
            # If no new data provided, make predictions on test data
            X_test = ml_results['X_test']
            y_test = ml_results['y_test']
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            logger.info(f"Predictions on test data - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            
            predictions = {
                'model_name': best_model_name,
                'test_predictions': {
                    'y_true': y_test,
                    'y_pred': y_pred,
                    'metrics': {
                        'r2': r2,
                        'rmse': rmse,
                        'mae': mae
                    }
                }
            }
        else:
            # Make predictions on new data
            feature_cols = ml_results['feature_names']
            
            # Ensure all required features are present
            missing_cols = [col for col in feature_cols if col not in new_data.columns]
            if missing_cols:
                logger.warning(f"Missing features in new data: {missing_cols}")
                
                # Try to engineer missing features
                for col in missing_cols:
                    if col == 'energy_squared' and 'energy' in new_data.columns:
                        new_data['energy_squared'] = new_data['energy'] ** 2
                    elif col == 'energy_cubed' and 'energy' in new_data.columns:
                        new_data['energy_cubed'] = new_data['energy'] ** 3
                    elif col == 'diameter_squared' and 'channel_diameter' in new_data.columns:
                        new_data['diameter_squared'] = new_data['channel_diameter'] ** 2
                    elif col == 'distance_squared' and 'detector_distance' in new_data.columns:
                        new_data['distance_squared'] = new_data['detector_distance'] ** 2
                    elif col == 'angle_squared' and 'detector_angle' in new_data.columns:
                        new_data['angle_squared'] = new_data['detector_angle'] ** 2
                    elif col == 'log_energy' and 'energy' in new_data.columns:
                        new_data['log_energy'] = np.log1p(new_data['energy'])
                    elif col == 'log_diameter' and 'channel_diameter' in new_data.columns:
                        new_data['log_diameter'] = np.log1p(new_data['channel_diameter'])
                    elif col == 'log_distance' and 'detector_distance' in new_data.columns:
                        new_data['log_distance'] = np.log1p(new_data['detector_distance'])
                    elif col == 'log_aspect_ratio' and 'aspect_ratio' in new_data.columns:
                        new_data['log_aspect_ratio'] = np.log1p(new_data['aspect_ratio'])
                    elif col == 'energy_x_diameter' and 'energy' in new_data.columns and 'channel_diameter' in new_data.columns:
                        new_data['energy_x_diameter'] = new_data['energy'] * new_data['channel_diameter']
                    elif col == 'energy_x_distance' and 'energy' in new_data.columns and 'detector_distance' in new_data.columns:
                        new_data['energy_x_distance'] = new_data['energy'] * new_data['detector_distance']
                    elif col == 'energy_x_angle' and 'energy' in new_data.columns and 'detector_angle' in new_data.columns:
                        new_data['energy_x_angle'] = new_data['energy'] * new_data['detector_angle']
                    elif col == 'diameter_x_distance' and 'channel_diameter' in new_data.columns and 'detector_distance' in new_data.columns:
                        new_data['diameter_x_distance'] = new_data['channel_diameter'] * new_data['detector_distance']
                
            # Check again for missing features
            missing_cols = [col for col in feature_cols if col not in new_data.columns]
            if missing_cols:
                logger.error(f"Still missing features after engineering: {missing_cols}")
                return None
            
            # Make predictions
            X_new = new_data[feature_cols]
            y_pred = best_model.predict(X_new)
            
            # Store predictions in the dataframe
            new_data['predicted_dose'] = y_pred
            
            logger.info(f"Made predictions on {len(new_data)} new data points")
            
            predictions = {
                'model_name': best_model_name,
                'new_predictions': {
                    'data': new_data,
                    'y_pred': y_pred
                }
            }
        
        return predictions

@timeit
def perform_ml_analysis(results):
    """
    Perform comprehensive machine learning analysis on simulation results.
    
    Parameters:
    -----------
    results : list
        List of simulation result dictionaries
    
    Returns:
    --------
    ml_results : dict
        Dictionary of machine learning results
    """
    with LogSection("Performing machine learning analysis"):
        # Prepare dataset
        dataset = prepare_ml_dataset(results)
        
        # Perform feature engineering
        enhanced_dataset, feature_importance = feature_engineering(dataset)
        
        # Train models
        model_results = train_ml_models(enhanced_dataset)
        
        # Optimize best models
        optimized_models = perform_model_optimization(enhanced_dataset)
        
        # Combine results
        ml_results = {
            'dataset': enhanced_dataset,
            'feature_importance': feature_importance,
            'models': model_results['models'],
            'best_model': model_results['best_model'],
            'X_test': model_results['X_test'],
            'y_test': model_results['y_test'],
            'feature_names': model_results['feature_names'],
            'optimized_models': optimized_models
        }
        
        # Create visualizations
        ml_results['figures'] = plot_ml_results(ml_results)
        
        # Make predictions on test data
        ml_results['predictions'] = make_dose_predictions(ml_results)
        
        return ml_results

if __name__ == '__main__':
    # Set up logging
    from logging_utils import setup_logger
    logger = setup_logger(log_file="ml_analysis.log")
    
    # Load simulation results
    result_file = DATA_DIR / "simulation_results.pickle"
    if result_file.exists():
        with open(result_file, 'rb') as f:
            results = pickle.load(f)
        
        logger.info(f"Loaded {len(results)} simulation results from {result_file}")
        
        # Perform ML analysis
        ml_results = perform_ml_analysis(results)
        
        # Save ML results
        ml_result_file = DATA_DIR / "ml_results.pickle"
        with open(ml_result_file, 'wb') as f:
            pickle.dump(ml_results, f)
        
        logger.info(f"Saved machine learning results to {ml_result_file}")
    else:
        logger.error(f"Simulation results file not found: {result_file}")



