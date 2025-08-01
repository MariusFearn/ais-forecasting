# Installation, Setup, and Quick Testing Guide

## 🚀 Environment Setup

### **Prerequisites**
- **Conda**: Anaconda or Miniconda installed
- **Git**: For repository cloning
- **Hardware**: Minimum 16GB RAM recommended (54GB optimal)

### **1. Repository Setup**
```bash
# Clone the repository
git clone <repository-url>
cd ais-forecasting

# Verify project structure
ls -la
```

### **2. Environment Creation**
```bash
# Create and activate ML environment
conda create -n ML python=3.9 -y
conda activate ML

# IMPORTANT: Always use ML environment, never base
# Add to your ~/.bashrc for convenience:
echo "alias ml='conda activate ML'" >> ~/.bashrc
```

### **3. Dependency Installation**
```bash
# Install Python packages
pip install -r requirements.txt

# Verify key packages are installed
python -c "import pandas, numpy, h3, duckdb, xgboost; print('✅ Core packages installed')"
python -c "import torch; print('✅ PyTorch installed')"
python -c "import folium, geopandas; print('✅ Visualization packages installed')"
```

### **4. GPU Setup (Optional but Recommended)**
```bash
# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check XGBoost GPU support
python -c "import xgboost as xgb; print(f'XGBoost version: {xgb.__version__}')"
```

---

## ⚡ Quick Testing

### **System Health Check**
```bash
# Activate environment
conda activate ML

# Run system validation
python scripts/test_system.py
```
**Expected Output:**
```
✅ Environment: ML environment active
✅ Dependencies: All packages installed
✅ Data: AIS data files found
✅ Hardware: GPU detected (if available)
✅ System: Ready for processing
```

### **Maritime Discovery Quick Test**
```bash
# Fast integration test (8-10 seconds)
python scripts/test_maritime_discovery.py
```
**Expected Output:**
```
✅ Phase 1: Import validation successful
✅ Phase 2: AIS data loading successful (14.5M+ records)
✅ Phase 3: H3 functionality working
✅ Phase 4: Trajectory concepts validated
✅ Phase 5: Processing pipeline ready
🎯 All systems operational - ready for maritime discovery
```

### **Small Dataset Test**
```bash
# Test with 3 vessels (8-10 seconds)
python scripts/maritime_discovery.py \
    --config config/maritime_discovery_test.yaml \
    --output-dir ./data/processed/test_discovery
```
**Expected Output:**
```
📊 AIS Records: 22,712
🚢 Vessels: 3
🛣️ Trajectories: 20
🏢 Terminals: 112
⏱️ Total time: 8.7 seconds
✅ Test completed successfully
```

---

## 🔧 Configuration Validation

### **Check Configuration Files**
```bash
# Verify configuration structure
python -c "
import yaml
with open('config/default.yaml') as f:
    config = yaml.safe_load(f)
print('✅ Base configuration loaded')
print(f'Data path: {config[\"data\"][\"raw_data_path\"]}')
"
```

### **Validate Data Access**
```bash
# Check raw data availability
python -c "
from pathlib import Path
data_path = Path('data/raw')
pkl_files = list(data_path.glob('*.pkl'))
print(f'✅ Found {len(pkl_files)} AIS data files')
for f in pkl_files[:3]:
    print(f'  - {f.name}')
"
```

---

## 🏃 Running the Full System

### **1. Machine Learning Pipeline**
```bash
# Train model with current best configuration
python scripts/train_enhanced_model.py \
    --config config/experiment_configs/comprehensive_h3_experiment.yaml
```

### **2. Maritime Discovery Pipeline**
```bash
# Production run (10 vessels)
python scripts/maritime_discovery.py \
    --config config/maritime_discovery.yaml \
    --output-dir ./data/processed/maritime_discovery
```

### **3. Model Evaluation**
```bash
# Evaluate trained model
python scripts/evaluate_model.py \
    --model-path data/models/final_models/xgboost_model.pkl \
    --test-data data/processed/training_sets/test_features.parquet
```

---

## 📊 Performance Benchmarks

### **Expected Performance (Typical Hardware)**
- **System Test**: < 5 seconds
- **Integration Test**: 8-10 seconds
- **Small Dataset (3 vessels)**: 8-10 seconds
- **Production Dataset (10 vessels)**: 15-30 seconds
- **Full ML Training**: 2-5 minutes (with GPU)

### **Hardware Requirements**
| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| RAM | 16 GB | 32 GB | 54 GB |
| CPU | 4 cores | 8 cores | 14+ threads |
| GPU | None | GTX 1060 | RTX 3080+ |
| Storage | 50 GB | 100 GB | 200 GB+ |

---

## 🔍 Troubleshooting

### **Common Issues**

#### **Issue 1: Environment Not Activated**
```bash
# Check current environment
conda info --envs
# Should show (ML) in prompt

# Solution: Always activate ML environment
conda activate ML
```

#### **Issue 2: Missing Dependencies**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Check specific packages
pip list | grep -E "(pandas|numpy|xgboost|h3)"
```

#### **Issue 3: Data Files Not Found**
```bash
# Check data directory
ls -la data/raw/
# Should show ais_cape_data_*.pkl files

# Download data if missing (contact project maintainer)
```

#### **Issue 4: GPU Not Detected**
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Install CUDA-enabled packages if needed
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### **Performance Issues**

#### **Slow Processing**
- Verify using ML environment (not base)
- Check available RAM (should have 16GB+ free)
- Monitor CPU usage (should use multiple cores)
- Enable GPU acceleration if available

#### **Memory Errors**
- Reduce dataset size in configuration
- Use `max_vessels: 5` for testing
- Close other applications to free memory
- Check swap space availability

---

## 📈 Performance Monitoring

### **During Execution**
```bash
# Monitor system resources
htop  # CPU and memory usage
nvidia-smi  # GPU usage (if available)

# Monitor log output for timing information
tail -f logs/maritime_discovery.log
```

### **Benchmark Comparison**
```bash
# Run DuckDB vs Pandas benchmark
python scripts/benchmark_duckdb.py

# Expected speedup: 10-50x for aggregations
```

---

## ✅ Success Verification

### **After Installation**
- [ ] ML environment activated successfully
- [ ] All dependencies installed without errors
- [ ] System test passes completely
- [ ] Integration test completes in <10 seconds
- [ ] Small dataset test produces valid results

### **After First Run**
- [ ] Output files generated in correct directories
- [ ] Log files show performance metrics
- [ ] Visualization files can be opened in browser
- [ ] No error messages in terminal output

### **Production Readiness**
- [ ] Full pipeline completes without errors
- [ ] Performance meets expected benchmarks
- [ ] Output quality validated
- [ ] System resources properly utilized
