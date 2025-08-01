"""Configuration management utilities for maritime discovery project."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration with validation.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML configuration: {e}")
        raise

def merge_experiment_config(base_config: Dict, experiment_config: Dict) -> Dict:
    """
    Merge experiment-specific settings with base configuration.
    
    Args:
        base_config: Base configuration dictionary
        experiment_config: Experiment-specific overrides
        
    Returns:
        Merged configuration dictionary
    """
    merged_config = base_config.copy()
    
    # Merge experiment parameters if they exist
    if 'parameters' in experiment_config:
        for section, params in experiment_config['parameters'].items():
            if section in merged_config:
                if isinstance(merged_config[section], dict) and isinstance(params, dict):
                    merged_config[section].update(params)
                else:
                    merged_config[section] = params
            else:
                merged_config[section] = params
    
    # Add experiment metadata
    merged_config['experiment_info'] = experiment_config.get('experiment', {})
    
    logger.info("Successfully merged experiment configuration with base config")
    return merged_config

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If required parameters are missing or invalid
    """
    required_sections = ['data', 'processing', 'terminals', 'routes', 'visualization']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate data section
    if 'input_files' not in config['data'] or not config['data']['input_files']:
        raise ValueError("No input files specified in data configuration")
    
    # Validate processing parameters
    processing = config['processing']
    if processing.get('h3_resolution', 0) < 1 or processing.get('h3_resolution', 0) > 15:
        raise ValueError("H3 resolution must be between 1 and 15")
    
    # Validate terminal parameters
    terminals = config['terminals']
    if terminals.get('min_visits', 0) < 1:
        raise ValueError("Minimum visits must be at least 1")
    
    if terminals.get('min_vessels', 0) < 1:
        raise ValueError("Minimum vessels must be at least 1")
    
    logger.info("Configuration validation passed")
    return True

def get_output_paths(config: Dict[str, Any], create_dirs: bool = True) -> Dict[str, Path]:
    """
    Get standardized output paths from configuration.
    
    Args:
        config: Configuration dictionary
        create_dirs: Whether to create output directories
        
    Returns:
        Dictionary mapping output types to Path objects
    """
    paths = {
        'terminals': Path(config['terminals']['output_path']),
        'routes': Path(config['routes']['output_path']),
        'visualization': Path(config['visualization']['output_path'])
    }
    
    if create_dirs:
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
    
    return paths
