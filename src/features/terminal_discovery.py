"""Terminal discovery algorithm extracted from successful maritime exploration."""

import pandas as pd
import numpy as np
import h3
from typing import Dict, List, Tuple, Optional, Set
from geopy.distance import geodesic
import logging
from collections import defaultdict, Counter
import time

from ..utils.logging_setup import get_logger, log_performance_metrics

logger = get_logger(__name__)

class TerminalDiscovery:
    """
    Maritime terminal discovery using vessel behavior patterns.
    
    Discovers terminals and ports by analyzing:
    1. Stationary periods (speed < threshold)
    2. Vessel convergence patterns
    3. Cargo status transitions
    4. Approach/departure behaviors
    """
    
    def __init__(self, config: Dict):
        """
        Initialize terminal discovery with configuration.
        
        Args:
            config: Configuration dictionary with discovery parameters
        """
        self.config = config
        self.stationary_speed_threshold = config.get('stationary_speed_threshold', 1.0)
        self.min_stationary_duration = config.get('min_stationary_duration_hours', 2.0)
        self.terminal_proximity_km = config.get('terminal_proximity_km', 5.0)
        self.min_vessels_for_terminal = config.get('min_vessels_for_terminal', 3)
        self.h3_resolution = config.get('h3_resolution', 8)  # Higher res for terminals
        
    def discover_terminals(self, ais_data: pd.DataFrame) -> pd.DataFrame:
        """
        Discover maritime terminals from AIS data.
        
        Args:
            ais_data: DataFrame with columns [imo, mdt, lat, lon, speed, draught, etc.]
            
        Returns:
            DataFrame with discovered terminals and their characteristics
        """
        logger.info("Starting terminal discovery from AIS data")
        start_time = time.time()
        
        if ais_data.empty:
            logger.warning("No AIS data provided for terminal discovery")
            return pd.DataFrame()
        
        # Step 1: Identify stationary periods
        stationary_periods = self._identify_stationary_periods(ais_data)
        
        if stationary_periods.empty:
            logger.warning("No stationary periods found for terminal discovery")
            return pd.DataFrame()
        
        # Step 2: Cluster stationary locations
        terminal_clusters = self._cluster_stationary_locations(stationary_periods)
        
        # Step 3: Analyze vessel patterns at each cluster
        terminals = self._analyze_terminal_characteristics(terminal_clusters, ais_data)
        
        # Step 4: Filter and validate terminals
        validated_terminals = self._validate_terminals(terminals)
        
        duration = time.time() - start_time
        log_performance_metrics(
            logger,
            "Terminal Discovery",
            duration,
            len(validated_terminals)
        )
        
        logger.info(f"Discovered {len(validated_terminals)} maritime terminals")
        return validated_terminals
    
    def _identify_stationary_periods(self, ais_data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify periods where vessels are stationary.
        
        Uses original column names: imo, mdt, lat, lon, speed
        """
        logger.info("Identifying stationary periods")
        
        stationary_records = []
        
        # Group by vessel (imo)
        for imo, vessel_data in ais_data.groupby('imo'):
            vessel_data = vessel_data.sort_values('mdt').reset_index(drop=True)
            
            # Find stationary records
            stationary_mask = vessel_data['speed'] <= self.stationary_speed_threshold
            stationary_vessel_data = vessel_data[stationary_mask].copy()
            
            if stationary_vessel_data.empty:
                continue
            
            # Group consecutive stationary periods
            stationary_vessel_data['time_diff'] = stationary_vessel_data['mdt'].diff()
            stationary_vessel_data['period_group'] = (
                stationary_vessel_data['time_diff'] > pd.Timedelta(hours=1)
            ).cumsum()
            
            # Analyze each stationary period
            for period_id, period_data in stationary_vessel_data.groupby('period_group'):
                if len(period_data) < 2:
                    continue
                
                duration_hours = (period_data['mdt'].max() - period_data['mdt'].min()).total_seconds() / 3600
                
                if duration_hours >= self.min_stationary_duration:
                    # Create stationary period record
                    center_lat = period_data['lat'].mean()
                    center_lon = period_data['lon'].mean()
                    
                    stationary_records.append({
                        'imo': imo,
                        'period_start': period_data['mdt'].min(),
                        'period_end': period_data['mdt'].max(),
                        'duration_hours': duration_hours,
                        'center_lat': center_lat,
                        'center_lon': center_lon,
                        'h3_cell': h3.geo_to_h3(center_lat, center_lon, self.h3_resolution),
                        'avg_draught': period_data.get('draught', pd.Series([10])).mean(),
                        'record_count': len(period_data)
                    })
        
        if not stationary_records:
            return pd.DataFrame()
        
        stationary_df = pd.DataFrame(stationary_records)
        logger.info(f"Found {len(stationary_df)} stationary periods from {ais_data['imo'].nunique()} vessels")
        
        return stationary_df
    
    def _cluster_stationary_locations(self, stationary_periods: pd.DataFrame) -> pd.DataFrame:
        """
        Cluster stationary locations to identify terminal candidates.
        """
        logger.info("Clustering stationary locations")
        
        if stationary_periods.empty:
            return pd.DataFrame()
        
        # Use H3 cells for initial clustering
        location_clusters = []
        
        for h3_cell, cell_periods in stationary_periods.groupby('h3_cell'):
            if len(cell_periods) < 2:  # Need multiple periods for a terminal
                continue
            
            # Get representative location
            center_lat = cell_periods['center_lat'].mean()
            center_lon = cell_periods['center_lon'].mean()
            
            # Check vessel diversity
            unique_vessels = cell_periods['imo'].nunique()
            
            if unique_vessels >= self.min_vessels_for_terminal:
                location_clusters.append({
                    'terminal_id': f"terminal_{h3_cell}",
                    'h3_cell': h3_cell,
                    'lat': center_lat,
                    'lon': center_lon,
                    'vessel_count': unique_vessels,
                    'period_count': len(cell_periods),
                    'total_duration_hours': cell_periods['duration_hours'].sum(),
                    'avg_duration_hours': cell_periods['duration_hours'].mean(),
                    'periods': cell_periods.to_dict('records')
                })
        
        if not location_clusters:
            return pd.DataFrame()
        
        clusters_df = pd.DataFrame(location_clusters)
        logger.info(f"Created {len(clusters_df)} terminal clusters")
        
        return clusters_df
    
    def _analyze_terminal_characteristics(self, terminal_clusters: pd.DataFrame, ais_data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze characteristics of each terminal cluster.
        """
        logger.info("Analyzing terminal characteristics")
        
        if terminal_clusters.empty:
            return pd.DataFrame()
        
        enhanced_terminals = []
        
        for _, terminal in terminal_clusters.iterrows():
            # Get all AIS data near this terminal
            terminal_vicinity = self._get_terminal_vicinity_data(
                ais_data, terminal['lat'], terminal['lon']
            )
            
            if terminal_vicinity.empty:
                continue
            
            # Analyze approach/departure patterns
            approach_patterns = self._analyze_approach_patterns(terminal_vicinity)
            
            # Analyze cargo patterns
            cargo_patterns = self._analyze_cargo_patterns(terminal_vicinity)
            
            # Analyze vessel types
            vessel_types = self._analyze_vessel_types(terminal_vicinity)
            
            # Enhance terminal data
            enhanced_terminal = terminal.to_dict()
            enhanced_terminal.update({
                'approach_count': approach_patterns['approach_count'],
                'departure_count': approach_patterns['departure_count'],
                'avg_approach_speed': approach_patterns['avg_approach_speed'],
                'avg_departure_speed': approach_patterns['avg_departure_speed'],
                'cargo_loaded_periods': cargo_patterns['loaded_periods'],
                'cargo_ballast_periods': cargo_patterns['ballast_periods'],
                'cargo_transition_count': cargo_patterns['transition_count'],
                'vessel_types': vessel_types['types'],
                'activity_score': self._calculate_activity_score(terminal_vicinity),
                'terminal_type': self._classify_terminal_type(cargo_patterns, vessel_types)
            })
            
            enhanced_terminals.append(enhanced_terminal)
        
        return pd.DataFrame(enhanced_terminals)
    
    def _get_terminal_vicinity_data(self, ais_data: pd.DataFrame, terminal_lat: float, terminal_lon: float) -> pd.DataFrame:
        """
        Get all AIS data within terminal vicinity.
        """
        # Simple distance-based filter
        vicinity_data = []
        
        for _, record in ais_data.iterrows():
            distance_km = geodesic(
                (terminal_lat, terminal_lon),
                (record['lat'], record['lon'])
            ).kilometers
            
            if distance_km <= self.terminal_proximity_km:
                vicinity_data.append(record)
        
        if not vicinity_data:
            return pd.DataFrame()
        
        return pd.DataFrame(vicinity_data)
    
    def _analyze_approach_patterns(self, vicinity_data: pd.DataFrame) -> Dict:
        """
        Analyze vessel approach and departure patterns.
        """
        patterns = {
            'approach_count': 0,
            'departure_count': 0,
            'avg_approach_speed': 0,
            'avg_departure_speed': 0
        }
        
        if vicinity_data.empty:
            return patterns
        
        # Group by vessel and analyze speed patterns
        approach_speeds = []
        departure_speeds = []
        
        for imo, vessel_data in vicinity_data.groupby('imo'):
            vessel_data = vessel_data.sort_values('mdt')
            
            if len(vessel_data) < 3:
                continue
            
            speeds = vessel_data['speed'].values
            
            # Detect approach pattern (high to low speed)
            for i in range(len(speeds) - 2):
                if speeds[i] > 8 and speeds[i+2] < 2:
                    patterns['approach_count'] += 1
                    approach_speeds.append(speeds[i])
                    break
            
            # Detect departure pattern (low to high speed)
            for i in range(len(speeds) - 2):
                if speeds[i] < 2 and speeds[i+2] > 8:
                    patterns['departure_count'] += 1
                    departure_speeds.append(speeds[i+2])
                    break
        
        # Calculate averages
        if approach_speeds:
            patterns['avg_approach_speed'] = np.mean(approach_speeds)
        if departure_speeds:
            patterns['avg_departure_speed'] = np.mean(departure_speeds)
        
        return patterns
    
    def _analyze_cargo_patterns(self, vicinity_data: pd.DataFrame) -> Dict:
        """
        Analyze cargo loading patterns at terminal.
        """
        patterns = {
            'loaded_periods': 0,
            'ballast_periods': 0,
            'transition_count': 0
        }
        
        if vicinity_data.empty or 'draught' not in vicinity_data.columns:
            return patterns
        
        # Analyze draught changes as proxy for cargo operations
        for imo, vessel_data in vicinity_data.groupby('imo'):
            vessel_data = vessel_data.sort_values('mdt')
            
            if len(vessel_data) < 2:
                continue
            
            draughts = vessel_data['draught'].values
            speeds = vessel_data['speed'].values
            
            # Count loaded/ballast periods
            for i, (draught, speed) in enumerate(zip(draughts, speeds)):
                if speed < self.stationary_speed_threshold:  # Stationary
                    if draught > 12:  # Loaded condition
                        patterns['loaded_periods'] += 1
                    else:  # Ballast condition
                        patterns['ballast_periods'] += 1
            
            # Count cargo transitions
            draught_changes = np.abs(np.diff(draughts))
            significant_changes = draught_changes > 2.0  # Significant draught change
            patterns['transition_count'] += np.sum(significant_changes)
        
        return patterns
    
    def _analyze_vessel_types(self, vicinity_data: pd.DataFrame) -> Dict:
        """
        Analyze types of vessels using the terminal.
        """
        vessel_info = {
            'types': [],
            'size_distribution': {}
        }
        
        if vicinity_data.empty:
            return vessel_info
        
        # Simple vessel classification based on available data
        vessel_types = []
        
        for imo, vessel_data in vicinity_data.groupby('imo'):
            # Basic classification based on draught and speed patterns
            avg_draught = vessel_data.get('draught', pd.Series([10])).mean()
            max_speed = vessel_data['speed'].max()
            
            if avg_draught > 15:
                vessel_type = 'large_cargo'
            elif avg_draught > 10:
                vessel_type = 'medium_cargo'
            else:
                vessel_type = 'small_cargo'
            
            vessel_types.append(vessel_type)
        
        # Count vessel types
        type_counts = Counter(vessel_types)
        vessel_info['types'] = list(type_counts.keys())
        vessel_info['size_distribution'] = dict(type_counts)
        
        return vessel_info
    
    def _calculate_activity_score(self, vicinity_data: pd.DataFrame) -> float:
        """
        Calculate terminal activity score.
        """
        if vicinity_data.empty:
            return 0.0
        
        # Score based on vessel count, record density, and time span
        vessel_count = vicinity_data['imo'].nunique()
        record_count = len(vicinity_data)
        
        if len(vicinity_data) < 2:
            return 0.0
        
        time_span_days = (vicinity_data['mdt'].max() - vicinity_data['mdt'].min()).days
        time_span_days = max(time_span_days, 1)  # Avoid division by zero
        
        # Activity score: vessels per day * record density
        activity_score = (vessel_count / time_span_days) * (record_count / vessel_count)
        
        return float(activity_score)
    
    def _classify_terminal_type(self, cargo_patterns: Dict, vessel_types: Dict) -> str:
        """
        Classify terminal type based on patterns.
        """
        loaded_periods = cargo_patterns['loaded_periods']
        ballast_periods = cargo_patterns['ballast_periods']
        transitions = cargo_patterns['transition_count']
        
        # Simple classification logic
        if transitions > 5:
            return 'loading_terminal'  # Active cargo operations
        elif loaded_periods > ballast_periods * 2:
            return 'export_terminal'   # Mostly loaded vessels
        elif ballast_periods > loaded_periods * 2:
            return 'import_terminal'   # Mostly ballast vessels
        else:
            return 'mixed_terminal'    # Mixed operations
    
    def _validate_terminals(self, terminals: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and filter terminal candidates.
        """
        if terminals.empty:
            return terminals
        
        # Filter based on activity and vessel diversity
        min_activity_score = self.config.get('min_activity_score', 1.0)
        min_vessel_count = self.config.get('min_vessels_for_terminal', 3)
        
        validated = terminals[
            (terminals['activity_score'] >= min_activity_score) &
            (terminals['vessel_count'] >= min_vessel_count)
        ].copy()
        
        # Sort by activity score
        validated = validated.sort_values('activity_score', ascending=False)
        
        logger.info(f"Validated {len(validated)} terminals out of {len(terminals)} candidates")
        
        return validated

def extract_terminal_locations(terminals_df: pd.DataFrame) -> List[Tuple[float, float, str]]:
    """
    Extract terminal locations for visualization or further analysis.
    
    Args:
        terminals_df: DataFrame from discover_terminals()
        
    Returns:
        List of (lat, lon, terminal_type) tuples
    """
    if terminals_df.empty:
        return []
    
    locations = []
    for _, terminal in terminals_df.iterrows():
        locations.append((
            terminal['lat'],
            terminal['lon'],
            terminal.get('terminal_type', 'unknown')
        ))
    
    return locations
