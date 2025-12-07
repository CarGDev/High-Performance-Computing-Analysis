# Matrix Alignment Prototype - HPC Performance Demonstration

This repository contains a standalone C implementation demonstrating memory alignment effects on high-performance computing performance. The prototype investigates performance variability issues related to memory alignment, specifically examining patterns described in OpenBLAS Issue #3879.

## Project Overview

This repository focuses on the practical implementation and benchmarking framework:

- **C Prototype**: Custom implementation demonstrating cache-blocked matrix multiplication with AVX SIMD optimizations
- **Memory Alignment Comparison**: Compares 64-byte cache-line aligned vs 16-byte aligned memory access patterns
- **Benchmarking Framework**: Automated scripts for performance testing and visualization
- **Performance Analysis**: Tools for measuring and visualizing alignment effects across different matrix sizes

## Project Structure

```text
.
├── matrix_alignment_prototype.c   # C implementation demonstrating alignment effects
├── Makefile                        # Build configuration for C prototype
├── generate_plots.py               # Python script for performance visualization
├── run_benchmark_sizes.sh         # Automated benchmarking script
├── run_all_tests.sh               # Complete test suite orchestrator
├── benchmark_results.csv          # Collected performance data (generated)
├── requirements.txt               # Python dependencies
└── assets/                        # Generated plots and figures
```

## Building and Running

### Prerequisites

- GCC compiler with AVX support
- Python 3 with matplotlib and numpy
- LaTeX distribution (for report compilation)
- Make utility

### Compiling the C Prototype

```bash
make
```

This will compile `matrix_alignment_prototype.c` with AVX optimizations enabled.

### Running Benchmarks

Run the complete test suite:

```bash
./run_all_tests.sh
```

Or run benchmarks for specific matrix sizes:

```bash
./run_benchmark_sizes.sh
```

For CSV output:

```bash
./matrix_alignment_prototype -s 1024 --csv
```

### Generating Visualizations

After running benchmarks, generate plots:

```bash
python3 generate_plots.py
```

Plots will be saved in the `assets/` directory.

## Key Features

### Memory Alignment Demonstration

The prototype demonstrates:

- **Aligned version**: Uses 64-byte cache-line aligned memory with `_mm256_load_ps` (aligned SIMD loads)
- **Misaligned version**: Uses 16-byte aligned memory with `_mm256_loadu_ps` (unaligned SIMD loads)
- **Cache-blocked algorithm**: Implements tiled matrix multiplication for optimal cache utilization
- **Performance variability analysis**: Measures and visualizes alignment effects across different matrix sizes

### Benchmarking Framework

The automated framework includes:

- Multiple matrix size testing (512, 1024, 1500, 2048)
- CSV data collection for reproducibility
- Python visualization generating multiple analysis plots
- Execution time, speedup ratio, variability, and GFLOPS metrics

## Results

The implementation demonstrates performance variability patterns consistent with OpenBLAS Issue #3879, showing:

- Peak variability of 6.6% at matrix size 512
- Size-dependent performance differences
- Architecture-sensitive alignment effects
- Reduced variability on modern hardware (Zen 3) compared to older architectures (Zen 2)

## Technical Details

### Memory Alignment Implementation

The prototype demonstrates two memory allocation strategies:

- **Aligned allocation**: Uses `posix_memalign()` to allocate memory aligned to 64-byte cache-line boundaries
- **Misaligned allocation**: Simulates C++ default 16-byte alignment by offsetting pointers from cache-line boundaries

### SIMD Optimizations

When compiled with AVX support, the implementation uses:

- `_mm256_load_ps()`: Aligned SIMD loads (faster, requires 32-byte alignment)
- `_mm256_loadu_ps()`: Unaligned SIMD loads (slower, works with any alignment)

### Cache-Blocked Algorithm

The matrix multiplication uses a tiled (cache-blocked) approach:

- Tile size: 64x64 elements (256 bytes for floats)
- Maximizes cache line utilization
- Reduces memory bandwidth requirements
- Enables better spatial locality

## Background

This implementation was developed to investigate performance variability patterns described in OpenBLAS Issue #3879, which reported up to 2x performance differences depending on memory alignment. The prototype demonstrates that:

- Performance variability is size-dependent and architecture-sensitive
- Modern CPUs (Zen 3) show reduced alignment sensitivity compared to older architectures (Zen 2)
- Proper cache-line alignment reduces performance unpredictability

## Related Work

This prototype is based on analysis of OpenBLAS Issue #3879, which documented performance variability in matrix multiplication operations due to memory alignment. The implementation demonstrates similar variability patterns while providing a standalone, reproducible example.

## License

This project is for educational and research purposes. Code implementations are provided for demonstration of HPC optimization principles related to memory alignment and SIMD vectorization.
