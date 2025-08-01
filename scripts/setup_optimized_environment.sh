#!/bin/bash
# GPU-Optimized Environment Setup for AIS Forecasting
# Optimized for Intel i7-12700K + RTX 3080 Ti + 54GB RAM

echo "🚀 Setting up optimized ML environment for AIS Forecasting..."

# Activate conda environment
source /home/marius/miniconda3/bin/activate ML

# Set CPU optimization environment variables
export OMP_NUM_THREADS=14
export MKL_NUM_THREADS=14
export NUMEXPR_NUM_THREADS=14

# Set GPU environment variables
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Verify setup
echo "✅ Environment configured:"
echo "   🖥️  CPU Threads: $OMP_NUM_THREADS"
echo "   🚀 GPU Device: $CUDA_VISIBLE_DEVICES"
echo "   💾 Memory optimized for 54GB RAM"

# Check GPU availability
python -c "
import torch
print(f'   🎯 PyTorch CUDA: {torch.cuda.is_available()}')
print(f'   🔥 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')
print(f'   💻 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo "🎯 Ready for maximum performance ML training!"
echo ""
echo "Usage examples:"
echo "  python scripts/train_h3_model.py --config comprehensive_h3_experiment"
echo "  python scripts/create_training_data.py --config massive_data_creation"
