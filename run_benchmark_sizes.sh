#!/bin/bash
# Benchmark script to test multiple matrix sizes
# Based on OpenBLAS Issue #3879 performance variability analysis

# Matrix sizes to test (from before the fix issue 3879.md)
SIZES=(512 1024 1500 2048)

# Output file for results
RESULTS_FILE="benchmark_results.csv"

echo "=========================================="
echo "HPC Alignment Benchmark - Multiple Sizes"
echo "=========================================="
echo ""

# Create results file with header
echo "Matrix_Size,Aligned_Time,Misaligned_Time,Speedup" > "$RESULTS_FILE"

# Build the prototype if needed
if [ ! -f "matrix_alignment_prototype" ]; then
    echo "Building prototype..."
    make clean
    make
    if [ $? -ne 0 ]; then
        echo "Error: Build failed"
        exit 1
    fi
fi

# Run benchmarks for each size
for SIZE in "${SIZES[@]}"; do
    echo "Testing matrix size: ${SIZE}x${SIZE}"
    echo "----------------------------------------"
    
    # Run the benchmark with specific size in CSV mode
    ./matrix_alignment_prototype -s "$SIZE" --csv >> "$RESULTS_FILE"
    
    if [ $? -ne 0 ]; then
        echo "Error: Benchmark failed for size $SIZE"
        exit 1
    fi
    
    echo ""
done

echo "=========================================="
echo "Benchmark complete!"
echo "Results saved to: $RESULTS_FILE"
echo "=========================================="

# Generate plots
if command -v python3 &> /dev/null; then
    echo ""
    echo "Generating plots..."
    python3 generate_plots.py
else
    echo ""
    echo "Python3 not found. Skipping plot generation."
    echo "Install Python3 and matplotlib to generate plots."
fi

