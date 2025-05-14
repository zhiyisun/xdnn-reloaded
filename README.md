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

## Building the Library

### Using CMake

1. Make sure you have CMake installed (version 3.10 or higher)

2. You can build the library using the provided build script:
   ```bash
   ./build.sh
   ```

   Or manually:
   ```bash
   mkdir -p build
   cd build
   cmake ..
   cmake --build .
   ```

3. This will generate both:
   - `libxdnn.so`: Shared library
   - `libxdnn.a`: Static library

### Build Options

You can customize the build by setting CMake variables:
- `-DCMAKE_BUILD_TYPE=Release|Debug`: Set build type (default: Release)
- `-DBUILD_SHARED_LIBS=ON|OFF`: Control whether to build shared library (default: ON)

Example:
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

### Installation

To install the libraries and headers:
```bash
cd build
cmake --install .
```

By default, this will install to `/usr/local/`. To specify an installation prefix:
```bash
cmake -DCMAKE_INSTALL_PREFIX=/your/install/path ..
cmake --install .
```