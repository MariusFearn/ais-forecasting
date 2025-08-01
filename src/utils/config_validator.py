"""
Configuration validation utilities for the shipping lane discovery system.
"""
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from ..constants import (
    DEFAULT_H3_RESOLUTION, DEFAULT_TIME_GAP_THRESHOLD_HOURS,
    MIN_JOURNEY_LENGTH_POINTS, DEFAULT_TERMINAL_EPS_DEGREES,
    DEFAULT_TERMINAL_MIN_SAMPLES, DEFAULT_ROUTE_EPS, DEFAULT_ROUTE_MIN_SAMPLES
)

class ConfigurationError(ValueError):
    """Raised when configuration is invalid."""
    pass

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize configuration dictionary.
    
    Args:
        config: Raw configuration dictionary
        
    Returns:
        Validated and normalized configuration
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    validated_config = config.copy()
    
    # Validate required sections
    required_sections = ['data', 'trajectory', 'terminals', 'routes']
    for section in required_sections:
        if section not in config:
            raise ConfigurationError(f"Missing required configuration section: {section}")
    
    # Validate data section
    validated_config['data'] = _validate_data_config(config['data'])
    
    # Validate trajectory section
    validated_config['trajectory'] = _validate_trajectory_config(config['trajectory'])
    
    # Validate terminals section
    validated_config['terminals'] = _validate_terminals_config(config['terminals'])
    
    # Validate routes section  
    validated_config['routes'] = _validate_routes_config(config['routes'])
    
    # Set H3 resolution with default
    validated_config['h3_resolution'] = config.get('h3_resolution', DEFAULT_H3_RESOLUTION)
    if not isinstance(validated_config['h3_resolution'], int) or validated_config['h3_resolution'] < 0:
        raise ConfigurationError("h3_resolution must be a non-negative integer")
    
    logging.info("Configuration validation successful")
    return validated_config

def _validate_data_config(data_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate data configuration section."""
    if 'raw_data_dir' not in data_config:
        raise ConfigurationError("Missing required data.raw_data_dir")
    
    data_dir = Path(data_config['raw_data_dir'])
    if not data_dir.exists():
        raise ConfigurationError(f"Data directory does not exist: {data_dir}")
    
    return data_config

def _validate_trajectory_config(trajectory_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate trajectory configuration section."""
    validated = trajectory_config.copy()
    
    # Set defaults
    validated.setdefault('time_gap_threshold_hours', DEFAULT_TIME_GAP_THRESHOLD_HOURS)
    validated.setdefault('min_journey_length', MIN_JOURNEY_LENGTH_POINTS)
    
    # Validate types and ranges
    if not isinstance(validated['time_gap_threshold_hours'], (int, float)) or validated['time_gap_threshold_hours'] <= 0:
        raise ConfigurationError("trajectory.time_gap_threshold_hours must be positive number")
    
    if not isinstance(validated['min_journey_length'], int) or validated['min_journey_length'] <= 0:
        raise ConfigurationError("trajectory.min_journey_length must be positive integer")
    
    # Validate output path
    if 'output_path' not in validated:
        raise ConfigurationError("Missing required trajectory.output_path")
    
    return validated

def _validate_terminals_config(terminals_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate terminals configuration section."""
    validated = terminals_config.copy()
    
    # Set defaults
    validated.setdefault('eps', DEFAULT_TERMINAL_EPS_DEGREES)
    validated.setdefault('min_samples', DEFAULT_TERMINAL_MIN_SAMPLES)
    
    # Validate clustering parameters
    if not isinstance(validated['eps'], (int, float)) or validated['eps'] <= 0:
        raise ConfigurationError("terminals.eps must be positive number")
    
    if not isinstance(validated['min_samples'], int) or validated['min_samples'] <= 0:
        raise ConfigurationError("terminals.min_samples must be positive integer")
    
    # Validate output path
    if 'output_path' not in validated:
        raise ConfigurationError("Missing required terminals.output_path")
    
    return validated

def _validate_routes_config(routes_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate routes configuration section."""
    validated = routes_config.copy()
    
    # Set defaults
    validated.setdefault('eps', DEFAULT_ROUTE_EPS)
    validated.setdefault('min_samples', DEFAULT_ROUTE_MIN_SAMPLES)
    
    # Validate clustering parameters
    if not isinstance(validated['eps'], (int, float)) or validated['eps'] <= 0:
        raise ConfigurationError("routes.eps must be positive number")
    
    if not isinstance(validated['min_samples'], int) or validated['min_samples'] <= 0:
        raise ConfigurationError("routes.min_samples must be positive integer")
    
    # Validate output path
    if 'output_path' not in validated:
        raise ConfigurationError("Missing required routes.output_path")
    
    return validated

def create_default_config(data_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    Create a default configuration dictionary.
    
    Args:
        data_dir: Path to raw AIS data directory
        output_dir: Path to output directory
        
    Returns:
        Default configuration dictionary
    """
    output_path = Path(output_dir)
    
    return {
        'data': {
            'raw_data_dir': str(data_dir)
        },
        'h3_resolution': DEFAULT_H3_RESOLUTION,
        'trajectory': {
            'time_gap_threshold_hours': DEFAULT_TIME_GAP_THRESHOLD_HOURS,
            'min_journey_length': MIN_JOURNEY_LENGTH_POINTS,
            'output_path': str(output_path / 'vessel_journeys.parquet')
        },
        'terminals': {
            'eps': DEFAULT_TERMINAL_EPS_DEGREES,
            'min_samples': DEFAULT_TERMINAL_MIN_SAMPLES,
            'output_path': str(output_path / 'maritime_terminals.gpkg')
        },
        'routes': {
            'eps': DEFAULT_ROUTE_EPS,
            'min_samples': DEFAULT_ROUTE_MIN_SAMPLES,
            'output_path': str(output_path / 'clustered_journeys.parquet')
        },
        'graph': {
            'output_path': str(output_path / 'route_graph.gpkg')
        },
        'visualization': {
            'output_path': str(output_path / 'shipping_lanes.html')
        }
    }
