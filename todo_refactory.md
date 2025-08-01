# 📋 Comprehensive Refactoring Plan for Global Maritime Discovery

## 🎯 Project Overview
This refactoring plan transforms the successful `discover_shipping_lanes_production.ipynb` notebook into a production-ready project structure following the established AIS forecasting architecture. The notebook successfully discovered 697 maritime terminals worldwide in 1.2 minutes after optimization.

## 🏗️ Current State Assessment

### ✅ What's Working Well
- **Performance**: Optimized from 21+ minutes to 1.2 minutes total runtime
- **Global Coverage**: Successfully processes 7.9M AIS records across 7 years
- **Terminal Discovery**: Identifies 697 real-world maritime terminals
- **Visualization**: Creates interactive global maps with 500-terminal performance limit
- **Data Outputs**: Generates GeoPackage, Parquet, and HTML files

### � Existing Infrastructure Analysis
**DISCOVERED: We already have optimized components that should be reused!**

✅ **Data Loading**: 
- `src/data/duckdb_engine.py` - Ultra-fast DuckDB-based data loading (10-50x speedup)
- `src/data/loader.py` - AISDataLoader with multi-year support and parquet optimization
- `data/raw/parquet/` - Optimized parquet files already available

✅ **Algorithm Functions**:
- `src/models/clustering.py` - Already contains `compute_dtw_distance_matrix()` and `cluster_routes()`
- Working column names: `imo`, `mdt`, `lat`, `lon` (no need to rename!)

### �🔧 Areas Requiring Refactoring
- **Configuration**: Mixed inline configuration with processing logic
- **Function Organization**: Extract terminal discovery from notebook to existing structure
- **Data Loading**: Use existing DuckDB engine instead of pickle files
- **Integration**: Connect existing functions with proper configuration system
- **Code Duplication**: Repeated processing patterns across cells
- **Error Handling**: Inconsistent error management
- **Logging**: Mixed print statements and logging calls
- **Testing**: No unit tests for complex algorithms
- **Documentation**: Limited function-level documentation

## 📁 Detailed Refactoring Strategy

### 1. Configuration Management
**Current State**: Configuration mixed throughout notebook cells
**Target State**: Clean YAML-based configuration system

#### 1.1 Create Core Configuration
**File**: `config/global_maritime_discovery.yaml`
```yaml
# Global Maritime Terminal Discovery Configuration
global_discovery:
  name: "Global Maritime Terminal Discovery"
  description: "Worldwide AIS data analysis for terminal and route discovery"
  
data:
  input_files:
    # Raw AIS data files (2018-2024)
    - "data/raw/ais_cape_data_2018.pkl"
    - "data/raw/ais_cape_data_2019.pkl"
    # ... etc
  date_range:
    start: "2018-01-01"
    end: "2024-12-31"
  
processing:
  memory_limit_gb: 32
  max_vessels_per_batch: 1000
  h3_resolution: 5
  min_journey_length: 5
  max_journey_length: 1000
  
terminals:
  min_visits: 10
  min_vessels: 3
  clustering:
    eps_km: 5.0
    min_samples: 3
  output_path: "data/processed/global_terminals.gpkg"
  
routes:
  max_routes_for_dtw: 150
  dtw_distance_threshold: 0.5
  clustering:
    eps: 0.3
    min_samples: 5
  sequence_limits:
    min_length: 5
    max_length: 200
  output_path: "data/processed/global_routes.parquet"
  
visualization:
  max_map_terminals: 500
  map_center: [20, 0]
  zoom_start: 2
  output_path: "visualizations/global_maritime_terminals.html"
  
performance:
  enable_progress_bars: true
  log_level: "INFO"
  save_checkpoints: true
```

#### 1.2 Move Experiment Variables
**File**: `experiments/global_maritime_discovery/experiment_config.yaml`
```yaml
# Experiment-specific settings
experiment:
  name: "global_terminal_discovery_production"
  version: "v1.0"
  date: "2024-01-XX"
  
parameters:
  # Override base config for this experiment
  terminals:
    min_visits: 10  # Experiment with different thresholds
  routes:
    max_routes_for_dtw: 200  # Higher for comprehensive analysis
  
results:
  terminals_discovered: 697
  processing_time_minutes: 1.2
  optimization_notes: "95% performance improvement achieved"
```

### 2. Source Code Organization

#### 2.1 Data Loading Module (UPDATED - Use Existing DuckDB)
**File**: `src/data/maritime_loader.py` 
**Status**: ✅ REUSE existing `src/data/loader.py` and `src/data/duckdb_engine.py`
```python
"""Global maritime data loading leveraging existing DuckDB optimization."""

from .duckdb_engine import DuckDBEngine
from .loader import AISDataLoader

def load_global_ais_data(
    years: List[str] = ["2018", "2019", "2020", "2021", "2022", "2023", "2024"],
    filters: Optional[Dict] = None,
    use_duckdb: bool = True
) -> pd.DataFrame:
    """
    Load global AIS data using existing optimized DuckDB engine.
    
    Uses original column names: imo, mdt, lat, lon (no renaming needed)
    Leverages parquet files for 10-50x speedup over pickle files.
    """
    loader = AISDataLoader(data_dir="data", use_duckdb=use_duckdb)
    return loader.load_multi_year_data_optimized(years, filters=filters)
```

#### 2.2 Trajectory Processing Module (UPDATED - Use Original Columns)
**File**: `src/features/trajectory_processor.py`
```python
"""Optimized trajectory processing using original AIS column names."""

import h3
import pandas as pd
import numpy as np
from typing import List, Tuple

def extract_vessel_trajectories(
    ais_data: pd.DataFrame,
    h3_resolution: int = 5,
    min_journey_length: int = 5
) -> pd.DataFrame:
    """
    Extract vessel trajectories with H3 spatial indexing.
    
    Uses original column names:
    - imo: vessel identifier (not mmsi)  
    - mdt: timestamp (not timestamp)
    - lat: latitude
    - lon: longitude
    """
```

#### 2.3 Terminal Discovery Module (NEW - Extract from Notebook)
**File**: `src/features/terminal_discovery.py`
```python
"""Maritime terminal discovery extracted from working notebook code."""

import pandas as pd
import geopandas as gpd
from sklearn.cluster import DBSCAN
from typing import Dict, Tuple

def discover_maritime_terminals(
    journey_data: pd.DataFrame,
    config: Dict
) -> gpd.GeoDataFrame:
    """
    Discover maritime terminals from journey endpoint data.
    
    Extracts the exact algorithm from discover_shipping_lanes_production.ipynb
    that successfully found 697 terminals worldwide.
    """
    
def extract_journey_endpoints(journeys: pd.DataFrame) -> pd.DataFrame:
    """Extract start and end points from journey data (from notebook)."""
    
def cluster_terminals(
    endpoints: pd.DataFrame,
    eps_km: float,
    min_samples: int
) -> pd.DataFrame:
    """Cluster endpoints into terminal locations (from notebook)."""
```

#### 2.4 Route Clustering Module (REUSE - Already Exists!)
**File**: `src/features/route_clustering.py` 
**Status**: ✅ REUSE existing `src/models/clustering.py`
```python
"""Route clustering using existing DTW implementation."""

from ..models.clustering import compute_dtw_distance_matrix, cluster_routes

# Re-export existing functions with maritime-specific interface
def cluster_shipping_routes(
    journey_data: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """
    Cluster shipping routes using existing DTW implementation.
    
    Leverages:
    - compute_dtw_distance_matrix() from src/models/clustering.py
    - cluster_routes() from src/models/clustering.py
    """

#### 2.5 Visualization Module
**File**: `src/visualization/maritime_map.py`
```python
"""Interactive maritime visualization with Folium."""

import folium
import geopandas as gpd
from pathlib import Path
from typing import Optional

def create_global_maritime_map(
    terminals: gpd.GeoDataFrame,
    config: Dict,
    max_terminals: int = 500
) -> folium.Map:
    """Create interactive global maritime terminal map."""
    
def add_terminals_to_map(
    map_obj: folium.Map,
    terminals: gpd.GeoDataFrame
) -> None:
    """Add terminal markers to Folium map with optimized performance."""
    
def create_terminal_popup(terminal: pd.Series) -> str:
    """Create HTML popup content for terminal marker."""
    
def add_map_legend(map_obj: folium.Map, terminal_count: int) -> None:
    """Add informative legend to the map."""
```

#### 2.6 Analysis Module
**File**: `src/analysis/maritime_analytics.py`
```python
"""Maritime data analysis and reporting functions."""

import pandas as pd
from typing import Dict, List

def generate_terminal_statistics(terminals: gpd.GeoDataFrame) -> Dict:
    """Generate comprehensive terminal statistics."""
    
def analyze_regional_distribution(terminals: gpd.GeoDataFrame) -> pd.DataFrame:
    """Analyze terminal distribution by geographic region."""
    
def create_summary_tables(
    terminals: gpd.GeoDataFrame,
    journeys: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """Create comprehensive data analysis tables."""
```

### 3. Script Organization

#### 3.1 Main Discovery Script
**File**: `scripts/discover_global_maritime_terminals.py`
```python
#!/usr/bin/env python3
"""
Global Maritime Terminal Discovery Pipeline

Executes the complete global terminal discovery workflow:
1. Load worldwide AIS data (2018-2024)
2. Process vessel trajectories
3. Discover maritime terminals
4. Cluster shipping routes
5. Generate interactive visualizations

Usage:
    python scripts/discover_global_maritime_terminals.py --config config/global_maritime_discovery.yaml
"""

import argparse
import logging
from pathlib import Path
import time

from src.data.maritime_loader import load_global_ais_data
from src.features.trajectory_processor import extract_vessel_trajectories
from src.features.terminal_discovery import discover_maritime_terminals
from src.features.route_clustering import compute_dtw_distance_matrix, cluster_routes
from src.visualization.maritime_map import create_global_maritime_map
from src.analysis.maritime_analytics import generate_terminal_statistics
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging

def main():
    """Execute the complete global maritime discovery pipeline."""
    
    # Setup and configuration
    args = parse_arguments()
    config = load_config(args.config_path)
    setup_logging(config['performance']['log_level'])
    
    logger = logging.getLogger(__name__)
    logger.info("Starting global maritime terminal discovery")
    
    start_time = time.time()
    
    try:
        # Phase 1: Data Loading
        logger.info("Phase 1: Loading global AIS data")
        ais_data = load_global_ais_data(
            config['data']['input_files'],
            date_range=(config['data']['date_range']['start'], 
                       config['data']['date_range']['end']),
            memory_limit_gb=config['processing']['memory_limit_gb']
        )
        
        # Phase 2: Trajectory Processing
        logger.info("Phase 2: Processing vessel trajectories")
        journeys = extract_vessel_trajectories(
            ais_data,
            h3_resolution=config['processing']['h3_resolution'],
            min_journey_length=config['processing']['min_journey_length']
        )
        
        # Phase 3: Terminal Discovery
        logger.info("Phase 3: Discovering maritime terminals")
        terminals = discover_maritime_terminals(journeys, config['terminals'])
        
        # Phase 4: Route Clustering
        logger.info("Phase 4: Clustering shipping routes")
        routes = cluster_routes_pipeline(journeys, config['routes'])
        
        # Phase 5: Visualization
        logger.info("Phase 5: Creating interactive visualization")
        maritime_map = create_global_maritime_map(
            terminals, 
            config['visualization']
        )
        
        # Phase 6: Analysis and Reporting
        logger.info("Phase 6: Generating analysis reports")
        statistics = generate_terminal_statistics(terminals)
        
        # Save results
        save_results(terminals, routes, maritime_map, statistics, config)
        
        total_time = time.time() - start_time
        logger.info(f"Global discovery completed in {total_time/60:.1f} minutes")
        logger.info(f"Discovered {len(terminals)} maritime terminals worldwide")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
```

#### 3.2 Analysis Script
**File**: `scripts/analyze_maritime_terminals.py`
```python
#!/usr/bin/env python3
"""
Maritime Terminal Analysis Script

Generates comprehensive analysis reports from discovered terminal data.
"""

# Focused analysis and reporting functionality
```

#### 3.3 Visualization Script
**File**: `scripts/create_maritime_visualization.py`
```python
#!/usr/bin/env python3
"""
Maritime Visualization Generator

Creates publication-ready visualizations from terminal and route data.
"""

# Standalone visualization generation
```

### 4. Testing Strategy

#### 4.1 Unit Tests
**File**: `tests/test_terminal_discovery.py`
```python
"""Unit tests for terminal discovery functionality."""

import pytest
import pandas as pd
import geopandas as gpd
from src.features.terminal_discovery import discover_maritime_terminals

class TestTerminalDiscovery:
    def test_endpoint_extraction(self):
        """Test journey endpoint extraction."""
        
    def test_terminal_clustering(self):
        """Test terminal clustering algorithm."""
        
    def test_terminal_validation(self):
        """Test terminal validation criteria."""
```

**File**: `tests/test_route_clustering.py`
```python
"""Unit tests for route clustering functionality."""

import pytest
import numpy as np
from src.features.route_clustering import compute_dtw_distance_matrix

class TestRouteClustering:
    def test_dtw_computation(self):
        """Test DTW distance matrix computation."""
        
    def test_route_optimization(self):
        """Test route selection optimization."""
```

#### 4.2 Integration Tests
**File**: `tests/test_maritime_pipeline.py`
```python
"""Integration tests for the complete maritime discovery pipeline."""

class TestMaritimePipeline:
    def test_complete_pipeline(self):
        """Test the complete discovery pipeline with sample data."""
        
    def test_performance_benchmarks(self):
        """Test performance meets optimization targets."""
```

### 5. Utility Modules

#### 5.1 Configuration Management
**File**: `src/utils/config.py`
```python
"""Configuration management utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration with validation."""
    
def merge_experiment_config(base_config: Dict, experiment_config: Dict) -> Dict:
    """Merge experiment-specific settings with base configuration."""
```

#### 5.2 Logging Setup
**File**: `src/utils/logging_setup.py`
```python
"""Centralized logging configuration."""

import logging
from pathlib import Path

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup consistent logging across the project."""
```

#### 5.3 Performance Monitoring
**File**: `src/utils/performance.py`
```python
"""Performance monitoring and optimization utilities."""

import time
import psutil
from typing import Optional

class PerformanceMonitor:
    """Monitor and log performance metrics during processing."""
    
def estimate_processing_time(data_size: int, algorithm: str) -> float:
    """Estimate processing time for different algorithms."""
```

### 6. Notebook Refactoring

#### 6.1 Clean Notebook Structure
**File**: `notebooks/global_maritime_discovery_clean.ipynb`

**Cell 1: Environment Setup**
```python
# Simple imports and setup - no complex logic
import sys
from pathlib import Path

# Add src to path
project_root = Path.cwd().parent
sys.path.append(str(project_root / "src"))

from src.utils.config import load_config
from src.utils.logging_setup import setup_logging
```

**Cell 2: Configuration Loading**
```python
# Load configuration from YAML
config = load_config("../config/global_maritime_discovery.yaml")
experiment_config = load_config("../experiments/global_maritime_discovery/experiment_config.yaml")

# Setup logging
setup_logging(config['performance']['log_level'])
```

**Cell 3: Execute Pipeline**
```python
# Execute main pipeline with clean function calls
from scripts.discover_global_maritime_terminals import main as run_discovery

results = run_discovery(config, experiment_config)
```

**Cell 4: Interactive Analysis**
```python
# Interactive analysis and visualization
from src.analysis.maritime_analytics import generate_terminal_statistics
from src.visualization.maritime_map import create_global_maritime_map

# Interactive exploration
statistics = generate_terminal_statistics(results['terminals'])
display(statistics['summary_table'])
```

### 7. Documentation Updates

#### 7.1 README.md Updates
Add section documenting the maritime discovery capabilities:

```markdown
## 🌍 Global Maritime Terminal Discovery

The project includes a comprehensive maritime intelligence system that discovers ports, terminals, and shipping lanes from worldwide AIS data.

### Quick Start - Maritime Discovery
```bash
# Activate ML environment
conda activate ML

# Run global terminal discovery
python scripts/discover_global_maritime_terminals.py --config config/global_maritime_discovery.yaml

# View results
open visualizations/global_maritime_terminals.html
```

### Results
- **697 Maritime Terminals Discovered** across all major shipping regions
- **Interactive Global Map** with performance-optimized rendering
- **Comprehensive Analysis** including regional distribution and terminal rankings
```

#### 7.2 Function Documentation
Ensure all functions include:
- **Purpose**: Clear description of what the function does
- **Parameters**: Type hints and descriptions
- **Returns**: Expected output format
- **Examples**: Usage examples for complex functions
- **Performance Notes**: Memory/time complexity for large-scale functions

## 🚀 Implementation Priority

### Phase 1: Core Infrastructure (Week 1)
1. ✅ Create configuration files (`global_maritime_discovery.yaml`)
2. ✅ Setup basic source code structure (`src/data/`, `src/features/`, etc.)
3. 🔄 **UPDATED**: Implement data loading using existing DuckDB engine (not pickle files)
4. 🔄 **UPDATED**: Adapt main script to use existing AISDataLoader and column names

### Phase 2: Algorithm Migration (Week 2)
1. 🔄 **UPDATED**: Adapt trajectory processing to use original columns (`imo`, `mdt`, `lat`, `lon`)
2. ✅ Extract terminal discovery algorithm from working notebook
3. ✅ **REUSE**: Link existing DTW functions from `src/models/clustering.py`
4. ✅ Add comprehensive error handling and logging

### Phase 3: Visualization & Analysis (Week 3)
1. ✅ Refactor visualization code to `maritime_map.py`
2. ✅ Create analysis module (`maritime_analytics.py`)
3. ✅ **REUSE**: Leverage existing performance monitoring from DuckDB engine
4. ✅ Clean up the notebook interface

### Phase 4: Testing & Documentation (Week 4)
1. ✅ Write unit tests for all major functions
2. ✅ Create integration tests for the complete pipeline
3. ✅ Update project documentation
4. ✅ Performance benchmarking and optimization validation

## 🎯 Success Criteria

### Performance Requirements
- ✅ **Total Runtime**: ≤ 2 minutes for global discovery (currently 1.2 minutes)
- ✅ **Memory Usage**: ≤ 32 GB for complete global dataset
- ✅ **Terminal Discovery**: ≥ 500 terminals worldwide (currently 697)
- ✅ **Visualization**: Interactive map loads in ≤ 30 seconds

### Code Quality Requirements
- ✅ **Modularity**: No function >100 lines, clear separation of concerns
- ✅ **Configuration**: All parameters in YAML, no hardcoded values
- ✅ **Error Handling**: Graceful failure with informative messages
- ✅ **Testing**: ≥80% test coverage for critical functions
- ✅ **Documentation**: Complete function docstrings and usage examples

### Project Structure Compliance
- ✅ **Scripts**: Simple execution logic, delegate to `src/` functions
- ✅ **Source**: All algorithms and reusable functions in `src/`
- ✅ **Configuration**: Experiment variables in `experiments/`
- ✅ **Notebooks**: Clean interface for interactive exploration
- ✅ **Data Paths**: Follow established `data/raw/`, `data/processed/` structure

## 📊 Expected Benefits

### For Development
- **Maintainability**: Clear separation between configuration, algorithms, and execution
- **Testability**: Isolated functions enable comprehensive unit testing
- **Reusability**: Maritime discovery components can be used in other projects
- **Scalability**: Modular structure supports future enhancements

### For Research
- **Reproducibility**: YAML configurations ensure consistent experiment parameters
- **Experimentation**: Easy to test different clustering parameters and algorithms
- **Collaboration**: Clear code structure facilitates team development
- **Publication**: Clean implementation supports academic publication

### For Production
- **Reliability**: Comprehensive error handling and logging
- **Performance**: Optimized algorithms with monitoring capabilities
- **Deployment**: Containerizable scripts with defined dependencies
- **Monitoring**: Built-in performance tracking and validation

---

## 📝 Implementation Notes

### Migration Strategy
1. **Incremental**: Migrate one module at a time while keeping notebook functional
2. **Validation**: Test each migrated component against original notebook results
3. **Performance**: Ensure refactored code maintains or improves performance
4. **Documentation**: Document changes and update examples as we proceed

### Risk Mitigation
- **Backup**: Keep original notebook as reference during migration
- **Testing**: Validate results match exactly before removing original code
- **Performance**: Monitor runtime and memory usage throughout refactoring
- **Configuration**: Ensure all hardcoded values are properly externalized

This refactoring plan transforms the successful prototype into a production-ready, maintainable, and scalable maritime discovery system while preserving all performance optimizations and capabilities.
