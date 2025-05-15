#!/bin/bash
set -e

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build tests
make -j$(nproc)

# Run tests
ctest --output-on-failure

echo "All tests completed"
