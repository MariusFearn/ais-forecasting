# Vessel H3 Trajectory Prediction - Simple ML Pipeline

## Project Goal: SIMPLE FIRST SUCCESS
Predict which H3 cell a vessel will visit next using machine learning. Start simple, then extend.

**Why this focus?** Get our first working ML model on vessel movement, then expand later.

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

## Phase 3: Simple ML Training Pipeline (Week 3) **[NEXT: READY TO START]**

### 3.1 Create Training Data **[READY TO START]**
- [ ] **Convert 65 features to ML datasets**
  - [ ] Input: Current vessel state (65 features)
  - [ ] Target: Next H3 cell vessel actually visited
  - [ ] Create proper train/validation/test splits
  - [ ] Script: `scripts/create_training_data.py`

### 3.2 Train Simple Classifier **[READY TO START]**
- [ ] **Random Forest Classifier**
  - [ ] 65 features → Next H3 cell prediction
  - [ ] Basic hyperparameter tuning
  - [ ] Save trained model
  - [ ] Script: `scripts/train_model.py`

### 3.3 Evaluate Performance **[READY TO START]**
- [ ] **Classification Metrics**
  - [ ] Accuracy: Did we predict the right cell?
  - [ ] Distance error: How far off in km?
  - [ ] Visualize predictions on map
  - [ ] Script: `scripts/evaluate_model.py`

**Phase 3 Success Goal**: >60% accuracy predicting next H3 cell, <15km average distance error

## Phase 4: Model Improvements (Week 4) **[AFTER PHASE 3 SUCCESS]**

### 4.1 Try Different Models
- [ ] XGBoost classifier
- [ ] Simple neural network
- [ ] Ensemble methods

### 4.2 Multi-step Prediction
- [ ] Predict next 3-5 H3 cells
- [ ] Sequence-to-sequence models

### 4.3 Advanced Features
- [ ] Add weather data
- [ ] Port information
- [ ] Fleet interactions

**Focus**: Get Phase 3 working first - simple next-cell prediction!

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
