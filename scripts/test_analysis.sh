#!/bin/bash

# Test script for academic dataset analysis
# Uses a subset of data for quick testing

echo "Testing academic dataset analysis script..."
echo "This will analyze a small subset of the data first."

# Create a temporary directory for testing
TEST_OUTPUT_DIR="/share/home/u14004/dhj/Falcon-main/dataset_analysis_test"
mkdir -p "$TEST_OUTPUT_DIR"

# Run the analysis
python /share/home/u14004/dhj/Falcon-main/scripts/analyze_dataset_academic.py \
  --train_dir /share/home/u14004/dhj/Falcon-main/data/collect_data/train \
  --val_dir /share/home/u14004/dhj/Falcon-main/data/collect_data/val \
  --output_dir "$TEST_OUTPUT_DIR"

echo ""
echo "Test complete! Check results in: $TEST_OUTPUT_DIR"
echo "PDF files generated:"
ls -lh "$TEST_OUTPUT_DIR"/*.pdf 2>/dev/null || echo "No PDF files found"

