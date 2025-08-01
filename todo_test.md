# Shipping Lane Discovery: Testing Plan

This document outlines the comprehensive testing strategy for the new shipping lane discovery system.

## 🎯 **Testing Objectives**

1. **Verify End-to-End Pipeline** - Ensure the complete workflow runs without errors
2. **Validate Data Quality** - Check that outputs are meaningful and correctly formatted
3. **Performance Testing** - Measure execution time and resource usage
4. **Result Validation** - Verify that discovered lanes match expected maritime patterns
5. **Edge Case Handling** - Test robustness with various data scenarios

---

## 📋 **Testing Phases**

### **Phase 1: Environment & Dependencies**
- [ ] Verify all dependencies are installed correctly (especially `dtaidistance>=2.3.10`)
- [ ] Test import statements for all new modules
- [ ] Validate configuration file loading
- [ ] Check data directory structure and permissions

#### **Key Files & Directories to Verify**
**Core Implementation Files:**
- [ ] `scripts/discover_lanes.py` - Main orchestration script
- [ ] `src/features/trajectory.py` - Journey segmentation and route processing
- [ ] `src/models/clustering.py` - Terminal and route clustering algorithms
- [ ] `src/visualization/lanes.py` - Interactive map generation
- [ ] `src/utils/metrics.py` - Validation metrics and logging
- [ ] `tests/test_lanes.py` - Unit tests

**Configuration Files:**
- [ ] `config/experiment_configs/lanes_discovery.yaml` - Main configuration
- [ ] `config/default.yaml` - Base configuration (inheritance)
- [ ] `requirements.txt` - Updated with dtaidistance dependency

**Data Directories:**
- [ ] `data/raw/` - AIS pickle files (*.pkl)
- [ ] `data/processed/` - Output directory structure
- [ ] `data/models/` - Model artifacts directory

**Output Directories:**
- [ ] `visualizations/` - Interactive HTML maps
- [ ] `data/processed/training_sets/` - Journey data outputs
- [ ] `data/processed/predictions/` - Route clustering results

**Documentation:**
- [ ] `make_route_plan.md` - Technical approach documentation
- [ ] `todo_lanes.md` - Implementation checklist
- [ ] `todo_test.md` - This testing plan

### **Phase 2: Unit Testing**
- [ ] Test trajectory segmentation functions
- [ ] Test H3 sequence conversion
- [ ] Test terminal clustering algorithms
- [ ] Test DTW distance computation
- [ ] Test route clustering functions
- [ ] Test visualization components

### **Phase 3: Integration Testing**
- [ ] Test Phase 1: Trajectory & Terminal Extraction
- [ ] Test Phase 2: Route Clustering  
- [ ] Test Phase 3: Route Graph Construction
- [ ] Test Phase 4: Validation & Visualization
- [ ] Test complete end-to-end pipeline

### **Phase 4: Data Quality Validation**
- [ ] Verify terminal locations are geographically reasonable
- [ ] Check that route clusters capture meaningful patterns
- [ ] Validate that terminal-route connections make sense
- [ ] Ensure visualization displays correctly

### **Phase 5: Performance & Scalability**
- [ ] Measure execution time for each phase
- [ ] Monitor memory usage during processing
- [ ] Test with different data subset sizes
- [ ] Identify bottlenecks and optimization opportunities

---

## 🧪 **Test Data Strategy**

### **Small Test Dataset (Quick Validation)**
- **Size**: 1-2 vessels, 1 month of data (~1,000 records)
- **Purpose**: Rapid development iteration and debugging
- **Expected Runtime**: < 5 minutes
- **Expected Outputs**: 2-5 terminals, 1-3 routes

### **Medium Test Dataset (Realistic Testing)**
- **Size**: 10-20 vessels, 6 months of data (~50,000 records)
- **Purpose**: Realistic performance and pattern validation
- **Expected Runtime**: 15-30 minutes
- **Expected Outputs**: 10-20 terminals, 5-15 routes

### **Large Test Dataset (Full Scale)**
- **Size**: All available vessels, full dataset
- **Purpose**: Production-scale validation
- **Expected Runtime**: 2-4 hours
- **Expected Outputs**: 50+ terminals, 20+ major shipping lanes

---

## ✅ **Success Criteria**

### **Functional Requirements**
- [ ] Pipeline completes without fatal errors
- [ ] All output files are generated correctly
- [ ] GeoPackage files can be opened in GIS software
- [ ] Interactive map displays properly in browser
- [ ] Validation metrics show reasonable clustering quality

### **Quality Requirements**
- [ ] Terminals are located near known ports/anchorages
- [ ] Routes connect logical terminal pairs
- [ ] High-traffic routes correspond to known shipping lanes
- [ ] Outlier detection identifies genuinely unusual journeys
- [ ] Silhouette score > 0.3 (indicating reasonable clustering)

### **Performance Requirements**
- [ ] Small dataset: < 5 minutes total runtime
- [ ] Medium dataset: < 30 minutes total runtime
- [ ] Memory usage stays within available RAM (54GB)
- [ ] No memory leaks during processing

---

## 🚨 **Risk Mitigation**

### **Known Potential Issues**
1. **DTW Computation Scaling** - O(n²) complexity may be slow for large datasets
2. **Memory Usage** - Distance matrices can become very large
3. **Geographic Edge Cases** - Vessels crossing dateline or poles
4. **Data Quality** - Missing timestamps, invalid coordinates
5. **Terminal Clustering** - May merge distinct nearby ports

### **Mitigation Strategies**
1. **Chunked Processing** - Process data in smaller batches if needed
2. **Memory Monitoring** - Add memory usage logging and warnings
3. **Geographic Validation** - Add coordinate bounds checking
4. **Data Cleaning** - Implement robust data validation steps
5. **Parameter Tuning** - Allow easy adjustment of clustering parameters

---

## 📊 **Test Metrics to Track**

### **Processing Metrics**
- Total runtime per phase
- Memory peak usage
- Number of journeys processed
- Data processing throughput (records/second)

### **Quality Metrics**
- Number of terminals discovered
- Number of route clusters found
- Outlier percentage
- Silhouette score
- Terminal coverage (geographic spread)
- Route connectivity (terminals linked)

### **Output Validation**
- GeoPackage file sizes and record counts
- Map rendering time and file size
- Coordinate validity and bounds
- Temporal range coverage

---

## 🔄 **Testing Workflow**

1. **Setup** - Prepare test environment and data
2. **Small Scale Test** - Quick validation with minimal data
3. **Debug & Fix** - Address any issues found
4. **Medium Scale Test** - Realistic performance evaluation
5. **Quality Review** - Validate geographic and maritime logic
6. **Large Scale Test** - Full production-scale validation
7. **Performance Optimization** - Address any bottlenecks
8. **Documentation** - Record findings and recommendations

---

## 📝 **Test Documentation**

Each test run should document:
- **Test Configuration** - Dataset size, parameters used
- **Runtime Performance** - Execution time, memory usage
- **Output Quality** - Number of terminals/routes, validation metrics
- **Issues Found** - Any errors, warnings, or unexpected behavior
- **Visual Validation** - Screenshots of generated maps
- **Recommendations** - Suggested improvements or parameter adjustments

This comprehensive testing approach will ensure our shipping lane discovery system is robust, accurate, and ready for production use.
