#!/bin/bash
# Complete test suite for HPC Alignment Benchmark
# Based on OpenBLAS Issue #3879 and Step.md setup instructions

set -e  # Exit on error

echo "=========================================="
echo "HPC Alignment Benchmark - Complete Test Suite"
echo "Based on OpenBLAS Issue #3879"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "matrix_alignment_prototype.c" ]; then
    echo "Error: matrix_alignment_prototype.c not found"
    echo "Please run this script from the project directory"
    exit 1
fi

# Step 1: Build the prototype
echo "Step 1: Building prototype..."
echo "----------------------------------------"
make clean
make
if [ $? -ne 0 ]; then
    echo "Error: Build failed"
    exit 1
fi
echo "✓ Build successful"
echo ""

# Step 2: Run benchmarks for all sizes
echo "Step 2: Running benchmarks for multiple sizes..."
echo "----------------------------------------"
./run_benchmark_sizes.sh
if [ $? -ne 0 ]; then
    echo "Error: Benchmark failed"
    exit 1
fi
echo ""

# Step 3: Display results
echo "Step 3: Benchmark Results Summary"
echo "----------------------------------------"
if [ -f "benchmark_results.csv" ]; then
    echo ""
    echo "Results:"
    column -t -s',' benchmark_results.csv
    echo ""
else
    echo "Warning: benchmark_results.csv not found"
fi

# Step 4: Check for plots
echo "Step 4: Generated Files"
echo "----------------------------------------"
if [ -f "alignment_benchmark_results.png" ]; then
    echo "✓ Main plot: alignment_benchmark_results.png"
fi
if [ -f "alignment_benchmark_execution_time.png" ]; then
    echo "✓ Execution time plot: alignment_benchmark_execution_time.png"
fi
if [ -f "alignment_benchmark_speedup.png" ]; then
    echo "✓ Speedup plot: alignment_benchmark_speedup.png"
fi
if [ -f "alignment_benchmark_variability.png" ]; then
    echo "✓ Variability plot: alignment_benchmark_variability.png"
fi
if [ -f "alignment_benchmark_gflops.png" ]; then
    echo "✓ GFLOPS plot: alignment_benchmark_gflops.png"
fi

echo ""
echo "=========================================="
echo "Test suite complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review benchmark_results.csv for detailed data"
echo "2. Check generated PNG plots for visualizations"
echo "3. Compare results with OpenBLAS Issue #3879 findings"
echo ""

