# H3 Feature Engineering Plan for BDI Prediction

## Concept Overview

Transform the entire maritime world into a "chess board" using H3 hexagonal grids where each cell contains aggregated shipping activity metrics. The hypothesis is that global patterns of maritime traffic across all regions can predict bulk dry cargo rates (BDI - Baltic Dry Index).

## Core Assumption
**Global maritime traffic patterns in H3 grid cells → BDI rate predictions**

The "chess board" analogy means:
- Each H3 cell = a chess square with specific characteristics (globally distributed)
- Ship movements = pieces moving across the world board
- Traffic density/patterns = global board state 
- BDI rates = outcome we want to predict from global maritime activity
- Some regions will be more important than others (major shipping lanes, ports, chokepoints)

## Implementation Plan

### Phase 1: Basic H3 Grid Setup

#### 1.1 H3 Resolution Selection
```python
# Test different resolutions for global maritime coverage
h3_resolutions = {
    3: "~7,179 km² per cell",    # Ocean basin scale
    4: "~1,770 km² per cell",    # Regional sea scale
    5: "~252 km² per cell",      # Major port region scale  
    6: "~36 km² per cell",       # Port area scale
    7: "~5.2 km² per cell",      # Harbor scale
    8: "~0.74 km² per cell"      # High resolution
}
```

**Decision Criteria:**
- Balance between global coverage and computational efficiency
- Ensure sufficient ships per cell for meaningful statistics across diverse regions
- Consider major shipping lanes, ports, and maritime chokepoints
- Account for regional importance differences (some cells will be much more active)

#### 1.2 Grid Creation
```python
def create_h3_grid(lat_min, lat_max, lon_min, lon_max, resolution):
    """Create H3 grid covering global maritime areas of interest"""
    # Generate all H3 cells in bounding box (can be global or regional)
    # Return list of H3 cell IDs
    # Note: Some regions will be more important than others for BDI prediction
```

### Phase 2: AIS Data Transformation

#### 2.1 Point-to-Cell Mapping
```python
def map_ais_to_h3(df_ais, h3_resolution):
    """Convert lat/lon positions to H3 cell IDs"""
    df_ais['h3_cell'] = df_ais.apply(
        lambda row: h3.geo_to_h3(row['lat'], row['lon'], h3_resolution), 
        axis=1
    )
    return df_ais
```

#### 2.2 Temporal Aggregation
```python
# Aggregate by time windows
time_windows = ['1H', '6H', '1D', '1W']  # Hourly to weekly

def aggregate_h3_features(df_ais, time_window, h3_resolution):
    """Aggregate AIS data by H3 cells and time windows"""
    features = df_ais.groupby(['h3_cell', pd.Grouper(key='mdt', freq=time_window)]).agg({
        'imo': 'nunique',                    # Unique ships
        'speed': ['mean', 'std', 'max'],     # Speed statistics
        'heading': ['mean', 'std'],          # Heading statistics  
        'draught': ['mean', 'std'],          # Load statistics
        'delta_distance': 'sum',             # Total distance
        'nav_status': lambda x: x.mode()[0] if not x.empty else None  # Dominant status
    })
    return features
```

### Phase 3: Feature Engineering

#### 3.1 Basic Cell Features
```python
def create_basic_h3_features(df_aggregated):
    """Create fundamental features for each H3 cell"""
    features = {
        # Traffic Volume
        'ship_count': 'Number of unique ships',
        'message_count': 'Number of AIS messages', 
        'traffic_density': 'Messages per km²',
        
        # Movement Characteristics
        'avg_speed': 'Average speed in cell',
        'speed_variance': 'Speed variation (congestion indicator)',
        'avg_heading': 'Average heading direction',
        'heading_variance': 'Heading dispersion (traffic pattern indicator)',
        
        # Load Indicators
        'avg_draught': 'Average ship draught (cargo load)',
        'draught_variance': 'Load variation',
        'heavy_ship_ratio': 'Ratio of heavily loaded ships',
        
        # Distance/Activity
        'total_distance': 'Total distance traveled in cell',
        'avg_distance_per_ship': 'Average distance per ship',
        
        # Operational Status
        'anchored_ratio': 'Percentage of anchored ships',
        'transit_ratio': 'Percentage of ships in transit',
        'port_activity_ratio': 'Percentage in port operations'
    }
    return features
```

#### 3.2 Spatial Relationship Features
```python
def create_spatial_h3_features(df_h3, h3_resolution):
    """Create features based on neighboring cells and regional importance"""
    
    for cell_id in df_h3['h3_cell'].unique():
        # Get neighboring cells
        neighbors = h3.k_ring(cell_id, 1)  # 1-ring neighbors
        
        # Spatial features
        features = {
            'neighbor_traffic_density': 'Avg traffic in neighboring cells',
            'traffic_gradient': 'Traffic density difference with neighbors', 
            'flow_convergence': 'Ships moving toward this cell',
            'flow_divergence': 'Ships moving away from this cell',
            'regional_centrality': 'Importance in global traffic network',
            'shipping_lane_proximity': 'Distance to major shipping lanes',
            'port_proximity': 'Distance to major global ports',
            'chokepoint_proximity': 'Distance to maritime chokepoints (Suez, Panama, etc.)'
        }
    
    return features
```

#### 3.3 Regional Importance Features
```python
def create_regional_importance_features(df_h3):
    """Weight features by regional importance for global BDI prediction"""
    
    # Define strategically important maritime regions
    important_regions = {
        'asia_pacific_routes': 'Major Asia-Pacific shipping lanes',
        'atlantic_routes': 'Trans-Atlantic shipping corridors', 
        'suez_canal_region': 'Suez Canal and Red Sea approaches',
        'panama_canal_region': 'Panama Canal and Caribbean approaches',
        'strait_of_hormuz': 'Persian Gulf chokepoint',
        'strait_of_malacca': 'Southeast Asia chokepoint',
        'major_bulk_ports': 'Key iron ore and coal loading ports',
        'major_discharge_ports': 'Key bulk cargo discharge ports'
    }
    
    return importance_weighted_features
```

#### 3.4 Temporal Pattern Features
```python
def create_temporal_h3_features(df_h3_timeseries):
    """Create time-based features for each H3 cell"""
    
    features = {
        # Trend Features
        'traffic_trend_7d': '7-day traffic trend',
        'speed_trend_7d': '7-day speed trend',
        'load_trend_7d': '7-day draught trend',
        
        # Seasonality Features  
        'hour_of_day_pattern': 'Hourly traffic patterns',
        'day_of_week_pattern': 'Weekly traffic patterns',
        'monthly_pattern': 'Monthly traffic patterns',
        
        # Lag Features
        'traffic_lag_1d': 'Previous day traffic',
        'traffic_lag_7d': 'Previous week traffic',
        'speed_lag_1d': 'Previous day average speed',
        
        # Rolling Statistics
        'traffic_ma_7d': '7-day moving average traffic',
        'traffic_volatility_7d': '7-day traffic volatility',
        'speed_ma_7d': '7-day moving average speed'
    }
    
    return features
```

### Phase 4: Target Variable Integration

#### 4.1 BDI Data Integration
```python
def integrate_bdi_data(df_h3_features, bdi_data):
    """Integrate Baltic Dry Index as target variable"""
    
    # Match BDI dates with H3 feature dates
    # Create target variables at different horizons
    targets = {
        'bdi_current': 'Current day BDI',
        'bdi_1d_ahead': '1-day ahead BDI', 
        'bdi_7d_ahead': '7-day ahead BDI',
        'bdi_30d_ahead': '30-day ahead BDI',
        'bdi_change_1d': '1-day BDI change',
        'bdi_change_7d': '7-day BDI change'
    }
    
    return df_features_with_targets
```

#### 4.2 Feature-Target Alignment
```python
def create_prediction_dataset(df_h3_features, prediction_horizon='7d'):
    """Create dataset for BDI prediction"""
    
    # Ensure no data leakage
    # Features: t, t-1, t-2, ... t-n
    # Target: t + prediction_horizon
    
    return X_features, y_target
```

### Phase 5: Validation and Testing

#### 5.1 Chess Board Visualization
```python
def visualize_h3_chess_board(df_h3, date, feature='traffic_density'):
    """Visualize the maritime 'chess board' at a specific time"""
    
    # Create map with H3 cells colored by feature values
    # Show Cape Town area with hexagonal grid
    # Color intensity = feature value
    
    return folium_map
```

#### 5.2 Feature Quality Checks
```python
def validate_h3_features(df_h3_features):
    """Validate quality of H3 features"""
    
    checks = {
        'global_coverage_check': 'Ensure major maritime regions covered',
        'temporal_continuity': 'Check for time gaps',
        'spatial_consistency': 'Verify neighbor relationships globally',
        'feature_distributions': 'Check for outliers/anomalies across regions',
        'regional_importance': 'Validate importance weighting',
        'correlation_analysis': 'Feature correlation with BDI across regions'
    }
    
    return validation_report
```

## Implementation Steps

### Step 1: Data Preparation (Week 1)
1. Load 2018 AIS data (global dataset)
2. Test H3 resolutions (3-7) on sample data from different regions
3. Choose optimal resolution based on:
   - Ship count per cell globally (target: 5-50 ships/cell/day average, but highly variable by region)
   - Computational efficiency for global processing
   - Coverage of major shipping lanes and ports
   - Regional importance considerations

### Step 2: Basic Feature Creation (Week 1-2)
1. Implement point-to-cell mapping globally
2. Create basic aggregation features
3. Test with 1-month of global data
4. Identify most active regions and shipping lanes
5. Validate feature quality across different maritime regions

### Step 3: Advanced Features (Week 2-3)
1. Add spatial relationship features globally
2. Implement regional importance weighting
3. Create features for major shipping chokepoints
4. Implement temporal pattern features
5. Create lag and rolling window features

### Step 4: Target Integration (Week 3)
1. Source BDI historical data
2. Align with H3 features temporally
3. Create multiple prediction horizons

### Step 5: Validation (Week 4)
1. Create chess board visualizations
2. Analyze feature-target correlations
3. Build simple baseline model
4. Validate approach feasibility

## Technical Considerations

### Performance Optimization
- Use vectorized operations for H3 calculations
- Implement chunking for large datasets
- Consider Dask for parallel processing
- Cache H3 neighbor relationships

### Memory Management
- Stream process large pickle files
- Use appropriate dtypes (int8 for small categories)
- Implement data compression for storage

### Scalability
- Design for multiple years of data
- Consider distributed computing for production
- Plan for real-time feature updates

## Success Metrics

### Phase 1 Success Criteria
1. **Global Coverage**: 90%+ of major shipping lanes mapped to H3 cells
2. **Regional Density**: Variable density but meaningful in key regions (5-500 ships/cell/day depending on location)
3. **Efficiency**: Process 1M global AIS records in <10 minutes
4. **Regional Identification**: Successfully identify and weight key maritime regions

### Phase 2 Success Criteria  
1. **Features**: 25+ meaningful H3 features created with regional importance weights
2. **Quality**: <10% missing values in global feature matrix
3. **Correlation**: At least 5 regionally-weighted features show >0.3 correlation with BDI
4. **Regional Coverage**: Features capture activity in all major bulk shipping regions

### Phase 3 Success Criteria
1. **Baseline Model**: Beat random prediction (R² > 0.15) on global BDI prediction
2. **Feature Importance**: Identify top 10 predictive regions/features
3. **Validation**: Consistent performance across different time periods and regions
4. **Regional Insights**: Model reveals which global regions most influence BDI

## Files to Create

```
src/features/
├── h3_features.py           # Main H3 feature engineering (global)
├── spatial_features.py     # Spatial relationship features  
├── temporal_features.py    # Time-based features
├── regional_importance.py  # Regional weighting and importance
└── target_preparation.py   # BDI target preparation

scripts/
├── create_h3_features.py   # Main global feature creation script
└── validate_h3_features.py # Global feature validation script

notebooks/
├── h3_exploration.ipynb    # H3 resolution testing (global)
├── chess_board_viz.ipynb   # Global maritime visualization
├── regional_analysis.ipynb # Regional importance analysis
└── feature_analysis.ipynb  # Feature-target analysis
```

## Next Immediate Steps

1. **Install H3 library**: `pip install h3`
2. **Test H3 resolution**: Run small test on 2018 global data sample
3. **Create basic mapping**: Implement global `map_ais_to_h3()` function
4. **Identify key regions**: Analyze which areas have highest traffic density
5. **Visualize results**: Create first global maritime chess board view
6. **Validate approach**: Check if concept works with global sample data

This plan provides a systematic approach to transform global AIS data into H3-based features for BDI prediction, treating the entire maritime world as a strategic chess board where traffic patterns across all regions determine global market outcomes. Some regions will naturally be more influential than others, and we'll capture this through regional importance weighting.
