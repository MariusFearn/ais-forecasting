# TODO: Vessel-Level H3 Feature Engineering

## Project Goal
Predict individual capsize vessel movements through H3 grid cells to understand maritime traffic patterns. Focus on vessel trajectory prediction rather than Baltic rate prediction.

## Phase 1: Data Foundation (Week 1) **[COMPLETE ✅]**

### 1.1 Data Exploration **[COMPLETE ✅]**
- [x] Load one capsize vessel pickle file (start with 2024 data) **[Done: vessel_exploration.ipynb]**
- [x] Analyze vessel count and data quality **[Done: vessel_exploration.ipynb]**
- [x] Examine temporal coverage (gaps, frequency) **[Done: vessel_exploration.ipynb]**
- [x] Document vessel characteristics (size distribution, routes) **[Done: vessel_exploration.ipynb]**

### 1.2 H3 Setup and Testing **[COMPLETE ✅]**
- [x] Install H3 library: `pip install h3` **[Done: in requirements.txt]**
- [x] Test H3 resolutions 4-6 on sample vessel data **[Done: vessel_exploration.ipynb]**
- [x] Analyze cells per vessel journey at different resolutions **[Done: vessel_exploration.ipynb]**
- [x] Choose resolution based on: **[Done: Resolution 5 chosen]**
  - Average 10-50 cells per typical voyage
  - Clear route patterns visible
  - Computational efficiency
- [x] Document resolution choice rationale **[Done: vessel_exploration.ipynb]**

### 1.3 Basic Vessel Tracking **[COMPLETE ✅]**
- [x] Implement `map_vessel_to_h3_sequence()` function **[Done: VesselH3Tracker class]**
- [x] Create vessel journey extraction for single vessel **[Done: VesselH3Tracker.convert_vessel_to_h3_sequence()]**
- [x] Validate data quality: remove GPS errors, outliers **[Done: VesselH3Tracker.validate_h3_sequence_quality()]**
- [x] Test on 5-10 individual vessel journeys **[Done: vessel_exploration.ipynb]**

## Phase 2: Individual Vessel Features (Week 2) **[COMPLETE ✅]**

### 2.1 Core Vessel Journey Features
- [x] **Current State Features** **[Done: VesselFeatureExtractor]**
  - [x] `current_h3_cell`: Current H3 position **[Done]**
  - [x] `current_speed`: Current speed over ground **[Done]**
  - [x] `current_heading`: Current course **[Done]**
  - [x] `time_in_current_cell`: Duration in current H3 cell **[Done]**

- [x] **Historical Sequence Features** **[Done: VesselFeatureExtractor]**
  - [x] `h3_sequence_last_24h`: Last 24 hours of H3 positions **[Done]**
  - [x] `speed_sequence_last_24h`: Speed history **[Done]**
  - [x] `heading_sequence_last_24h`: Course history **[Done]**
  - [x] `cells_visited_24h`: Number of unique cells visited **[Done]**

### 2.2 Movement Pattern Features
- [x] **Direction and Speed Patterns** **[Done: VesselFeatureExtractor]**
  - [x] `avg_speed_6h/12h/24h`: Moving average speeds **[Done]**
  - [x] `speed_trend`: Accelerating/decelerating pattern **[Done]**
  - [x] `heading_consistency`: Course change frequency **[Done]**
  - [x] `movement_efficiency`: Direct distance vs actual path **[Done]**

### 2.2 Movement Pattern Features **[COMPLETE ✅]**
- [x] **Direction and Speed Patterns** **[Done: VesselFeatureExtractor]**
  - [x] `avg_speed_6h/12h/24h`: Moving average speeds **[Done]**
  - [x] `speed_trend`: Accelerating/decelerating pattern **[Done]**
  - [x] `heading_consistency`: Course change frequency **[Done]**
  - [x] `movement_efficiency`: Direct distance vs actual path **[Done]**

- [x] **Journey Characteristics** **[Done: VesselFeatureExtractor]**
  - [x] `cells_from_last_port`: Distance from last port departure **[Done: distance_from_start]**
  - [x] `journey_duration`: Time since last port **[Done: total_journey_time]**
  - [x] `estimated_destination`: Most likely destination based on trajectory **[Done: basic implementation]**
  - [x] `progress_to_destination`: Estimated journey completion % **[Done: journey_phase]**

### 2.3 Contextual Features **[COMPLETE ✅]**
- [x] **Geographic Context** **[Done: VesselFeatureExtractor]**
  - [x] `distance_to_nearest_port`: Distance to closest major port **[Done: estimated implementation]**
  - [x] `in_shipping_lane`: Boolean for major shipping route **[Done: likely_shipping_lane]**
  - [x] `water_depth`: Ocean depth at current position **[Done: estimated_water_depth]**
  - [x] `coastal_proximity`: Distance from coastline **[Done: estimated_coastal_proximity]**

- [x] **Operational Context** **[Done: VesselFeatureExtractor]**
  - [x] `likely_cargo_status`: 'loaded', 'ballast', 'loading', 'discharging' **[Done]**
  - [x] `draught_change_indicator`: Recent draught changes **[Done]**
  - [x] `port_approach_behavior`: Speed/heading patterns near ports **[Done]**
  - [x] `anchorage_time`: Time spent stationary **[Done]**

### 2.4 Advanced Features **[COMPLETE ✅]**
- [x] **Lag Features** **[Done: VesselFeatureExtractor]**
  - [x] Historical values at 1h, 6h, 12h intervals **[Done]**
- [x] **Rolling Statistics** **[Done: VesselFeatureExtractor]**
  - [x] 6h and 12h windows with mean, std, min, max **[Done]**
- [x] **Comprehensive Feature Set: 65 total features extracted** **[Done]**

**Phase 2 Result: Successfully extracted 65 comprehensive vessel features ready for sequence modeling!**

## Phase 3: Sequence Processing Pipeline (Week 3) **[NEXT: READY TO START]**

### 3.1 Data Preprocessing **[READY TO START]**
- [x] Create `VesselH3Tracker` class **[Done: src/features/vessel_h3_tracker.py]**
- [x] Implement journey segmentation (port-to-port) **[Done: VesselH3Tracker]**
- [x] Handle data quality issues: **[Done: VesselH3Tracker]**
  - [x] GPS coordinate errors **[Done]**
  - [x] Missing timestamps **[Done]**
  - [x] Duplicate records **[Done]**
  - [x] Speed/course outliers **[Done]**

### 3.2 Feature Engineering Pipeline **[READY TO START]**
- [x] Create `VesselFeatureExtractor` class **[Done: src/features/vessel_features.py]**
- [x] Implement sliding window feature extraction **[Done: VesselFeatureExtractor]**
- [x] Add lag features (1h, 6h, 12h, 24h) **[Done: VesselFeatureExtractor]**
- [x] Create rolling statistics (mean, std, trend) **[Done: VesselFeatureExtractor]**

### 3.3 Sequence Dataset Creation **[TODO: NEXT STEP]**
- [ ] Create training sequences for ML models
- [ ] Implement proper temporal splits (no data leakage)
- [ ] Generate input-target pairs:
  - Input: Last N H3 positions + features
  - Target: Next H3 position(s)
- [ ] Save processed sequences in efficient format

## Phase 4: Model Development (Week 4)

### 4.1 Baseline Models
- [ ] **Simple Persistence Model**
  - [ ] Predict next cell = current cell
  - [ ] Predict based on current heading/speed
  - [ ] Calculate baseline accuracy metrics

- [ ] **Rule-Based Model**
  - [ ] Great circle route prediction
  - [ ] Port-destination routing
  - [ ] Speed-based time estimation

### 4.2 Machine Learning Models
- [ ] **Sequence Models**
  - [ ] LSTM for H3 sequence prediction
  - [ ] Transformer model for vessel trajectories
  - [ ] Test different sequence lengths (6h, 12h, 24h)

- [ ] **Multi-step Prediction**
  - [ ] 1-step: Next H3 cell (1-6 hours)
  - [ ] Multi-step: Next 4 H3 cells (24 hours)
  - [ ] Journey completion: Predict destination

### 4.3 Model Evaluation
- [ ] **Accuracy Metrics**
  - [ ] Next cell prediction accuracy
  - [ ] Distance error (km from actual position)
  - [ ] Route similarity metrics
  - [ ] Destination prediction accuracy

- [ ] **Validation Strategy**
  - [ ] Time-based splits (train on 2018-2022, test on 2023-2024)
  - [ ] Vessel-based splits (train on 80% vessels, test on 20%)
  - [ ] Route-based validation (different trade routes)

## Phase 5: Analysis and Insights (Week 5)

### 5.1 Model Interpretation
- [ ] **Feature Importance Analysis**
  - [ ] Which features best predict next position?
  - [ ] How much does history length matter?
  - [ ] Most predictive contextual features

- [ ] **Trajectory Analysis**
  - [ ] Common route patterns in H3 space
  - [ ] Seasonal route variations
  - [ ] Port approach/departure patterns
  - [ ] Speed optimization zones

### 5.2 Visualization and Validation
- [ ] **Interactive Maps**
  - [ ] Actual vs predicted vessel tracks
  - [ ] H3 grid overlay with predictions
  - [ ] Route probability heatmaps
  - [ ] Animation of vessel movements

- [ ] **Performance Analysis**
  - [ ] Accuracy by route type
  - [ ] Performance in different ocean regions
  - [ ] Prediction accuracy vs forecast horizon
  - [ ] Error patterns and failure modes

## Implementation Files Structure

```
src/features/
├── vessel_h3_tracker.py      # Core vessel tracking logic
├── vessel_features.py        # Feature engineering for vessels
├── sequence_processor.py     # Sequence data preparation
└── trajectory_features.py    # Advanced trajectory features

src/models/
├── vessel_sequence_model.py  # LSTM/Transformer for vessel prediction
├── baseline_models.py        # Simple baseline predictors
└── ensemble_model.py         # Combined prediction models

scripts/
├── process_vessel_sequences.py  # Main feature creation script
├── train_vessel_model.py        # Model training pipeline
└── evaluate_vessel_model.py     # Model evaluation and analysis

notebooks/
├── vessel_exploration.ipynb     # Initial data exploration
├── h3_resolution_analysis.ipynb # H3 resolution selection
├── vessel_trajectory_viz.ipynb  # Visualization development
└── model_performance_analysis.ipynb # Results analysis
```

## Success Criteria

### Phase 1 Success (Week 1)
- [ ] Successfully track 10+ individual vessel journeys in H3 space
- [ ] Choose optimal H3 resolution with clear rationale
- [ ] Validate data quality and processing pipeline

### Phase 2 Success (Week 2)
- [ ] Extract 20+ meaningful features per vessel per time step
- [ ] Create clean sequence datasets for multiple vessels
- [ ] Validate feature quality and completeness

### Phase 3 Success (Week 3)  
- [ ] Process full year of capsize vessel data into H3 sequences
- [ ] Create training/validation datasets with proper temporal splits
- [ ] Implement efficient data loading for model training

### Phase 4 Success (Week 4)
- [ ] Train functioning sequence model for vessel prediction
- [ ] Achieve >50% accuracy for next H3 cell prediction
- [ ] Beat baseline models significantly

### Phase 5 Success (Week 5)
- [ ] Generate meaningful insights about capsize vessel behavior
- [ ] Create compelling visualizations of predicted vs actual tracks
- [ ] Document which features and patterns are most predictive

## Next Immediate Actions

1. **Start Today**: Load one pickle file and examine 1-2 vessel journeys manually
2. **This Week**: Implement basic H3 mapping and test different resolutions
3. **Next Week**: Build feature extraction pipeline for vessel sequences
4. **Week 3**: Create training datasets and implement baseline models
5. **Week 4**: Train and evaluate sequence prediction models

## Key Questions to Answer

- [ ] What H3 resolution gives best balance of route detail vs computational efficiency?
- [ ] How much historical data is needed for accurate next-position prediction?
- [ ] Which vessel features are most predictive of route decisions?
- [ ] Can we predict vessel destinations from early journey segments?
- [ ] How does prediction accuracy vary by route type and ocean region?
- [ ] What are the failure modes and when do vessels behave unpredictably?

## Final Goal

**Build a system that can predict where individual capsize vessels will move next in H3 space, with >70% accuracy for 6-hour predictions and meaningful insights about maritime traffic patterns.**

This focused approach prioritizes vessel behavior understanding over immediate rate prediction, building foundation knowledge that could later support market forecasting if desired.
