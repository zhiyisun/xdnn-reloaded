# xdnn-reloaded

## Overview

This project is an implementation based on the header files defined in Intel's [xFasterTransformer](https://github.com/intel/xFasterTransformer)'s xdnn library (version 1.5.6). 

Since xdnn is not open-sourced, this project reimplements the functionality defined in the headers from:
https://github.com/intel/xFasterTransformer/releases/download/IntrinsicGemm/xdnn_v1.5.6.tar.gz

## Contents

The implementation includes various optimized GEMM operations supporting different data types:
- SGEMM (Single precision)
- HGEMM (Half precision)
- BGEMM (BFloat16)
- AMX SGEMM (Advanced Matrix Extensions)

Additional utilities include:
- Softmax implementation
- Transpose operations
- Various data type conversions (float16, bfloat16, fp8, uint4, etc.)

## Building

### Prerequisites

- CMake 3.10 or higher
- C++17 compatible compiler (GCC or Clang recommended)
- AVX2 and FMA instruction set support

### Build Options

The build system supports two main workflows:

1. **Building both library and tests** (default)
2. **Building tests only** (when the library is already built)

### Using the Consolidated Build Script

We provide a unified script (`xdnn.sh`) that handles all build, test, and cleanup operations:

```bash
./xdnn.sh [COMMAND] [OPTIONS]
```

#### Commands:

- `lib` - Build only the implementation libraries (static and shared)
- `test` - Build and run tests using existing libraries
- `all` - Build libraries and tests, then run tests
- `clean` - Clean build artifacts
- `help` - Show help message

#### Options:

- `--no-run` - For 'test' and 'all' commands: build tests without running them
- `--verbose` - For 'test' and 'all' commands: show verbose test output
- `--deep` - For 'clean' command: remove libraries in addition to build directories
- `-j, --jobs N` - Set number of parallel build jobs (default: auto)

### Common Build Scenarios

#### 1. Build everything and run tests

```bash
./xdnn.sh all
```

This will:
- Build both the static and shared libraries (`libxdnn_static.a` and `libxdnn.so`)
- Build all test executables
- Run the tests

#### 2. Build only the library

```bash
./xdnn.sh lib
```

This will build just the libraries without building or running any tests.

#### 3. Build and run tests only (assuming libraries exist)

```bash
./xdnn.sh test
```

This will:
- Skip building the library
- Build test executables
- Run the tests

#### 4. Clean build artifacts

```bash
./xdnn.sh clean
```

This will:
- Remove all build directories
- Remove generated library files (if backups exist)

For a deeper clean that removes all library files regardless of backups:

```bash
./xdnn.sh clean --deep
```