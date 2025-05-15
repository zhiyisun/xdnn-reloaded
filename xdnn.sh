#!/bin/bash
# xdnn-reloaded unified build script

set -e

# Help message
show_help() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  lib           Build only the implementation libraries (static and shared)"
    echo "  test          Build and run tests using existing libraries"
    echo "  all           Build libraries and tests, then run tests"
    echo "  clean         Clean build artifacts and generated library files"
    echo "  help          Show this help message"
    echo ""
    echo "Options:"
    echo "  --no-run      For 'test' and 'all' commands: build tests without running them"
    echo "  --verbose     For 'test' and 'all' commands: show verbose test output"
    echo "  --deep        For 'clean' command: force removal of all library files, even without backups"
    echo "  -j, --jobs N  Set number of parallel build jobs (default: auto)"
    echo ""
    echo "Examples:"
    echo "  $0 lib                Build only the libraries"
    echo "  $0 test               Build and run tests using existing libraries"
    echo "  $0 all                Build libraries and tests, then run tests"
    echo "  $0 test --no-run      Build tests without running them"
    echo "  $0 clean              Clean build directories and generated libraries"
    echo "  $0 clean --deep       Force removal of all library files"
}

# Check command line parameters
if [ $# -lt 1 ]; then
    show_help
    exit 1
fi

# Parse command
COMMAND=$1
shift

# Default values
RUN_TESTS=true
VERBOSE=false
DEEP_CLEAN=false
BUILD_JOBS=$(nproc)
CHECK_EXISTING_LIBS=true

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-run)
            RUN_TESTS=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --deep)
            DEEP_CLEAN=true
            shift
            ;;
        -j|--jobs)
            if [[ $2 =~ ^[0-9]+$ ]]; then
                BUILD_JOBS=$2
                shift 2
            else
                echo "Error: --jobs requires a numeric argument"
                exit 1
            fi
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Project directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MAIN_BUILD_DIR="${SCRIPT_DIR}/build"
TESTS_BUILD_DIR="${SCRIPT_DIR}/tests/build"

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Check if libraries exist
check_libraries() {
    if [ ! -f "${SCRIPT_DIR}/libxdnn.so" ] || [ ! -f "${SCRIPT_DIR}/libxdnn_static.a" ]; then
        echo -e "${RED}ERROR:${NC} XDNN libraries not found."
        if [ "$CHECK_EXISTING_LIBS" = true ]; then
            echo "Please build the libraries first with: $0 lib"
            exit 1
        fi
    else
        echo -e "${GREEN}Found${NC} XDNN libraries in project directory"
    fi
}

# Build the implementation libraries
build_libraries() {
    echo -e "${GREEN}Building XDNN libraries...${NC}"
    
    mkdir -p "${MAIN_BUILD_DIR}"
    cd "${MAIN_BUILD_DIR}"
    
    # Configure with CMake
    cmake -DBUILD_XDNN_LIBRARY=ON ..
    
    # Build just the library targets
    make -j${BUILD_JOBS} xdnn xdnn_static
    
    cd "${SCRIPT_DIR}"
    
    if [ -f "${SCRIPT_DIR}/libxdnn.so" ] && [ -f "${SCRIPT_DIR}/libxdnn_static.a" ]; then
        echo -e "${GREEN}Libraries built successfully:${NC}"
        echo "  - ${SCRIPT_DIR}/libxdnn.so (shared library)"
        echo "  - ${SCRIPT_DIR}/libxdnn_static.a (static library)"
    else
        echo -e "${RED}Error:${NC} Failed to build libraries"
        exit 1
    fi
}

# Build the tests
build_tests() {
    echo -e "${GREEN}Building tests...${NC}"
    
    # Check if libraries exist first
    check_libraries
    
    # Build the tests
    mkdir -p "${MAIN_BUILD_DIR}"
    cd "${MAIN_BUILD_DIR}"
    
    # Configure with CMake - WITHOUT rebuilding the library
    cmake -DBUILD_XDNN_LIBRARY=OFF ..
    
    # Build the test targets
    make -j${BUILD_JOBS}
    
    cd "${SCRIPT_DIR}"
    
    echo -e "${GREEN}Tests built successfully${NC}"
}

# Run the tests
run_tests() {
    # Find where tests are located
    TEST_DIR=""
    if [ -d "${MAIN_BUILD_DIR}/tests" ]; then
        TEST_DIR="${MAIN_BUILD_DIR}/tests"
    elif [ -d "${TESTS_BUILD_DIR}" ]; then
        TEST_DIR="${TESTS_BUILD_DIR}"
    else
        echo -e "${RED}ERROR:${NC} No test build directory found!"
        echo "Please build the tests first with: $0 test --no-run"
        exit 1
    fi
    
    echo -e "${GREEN}Running tests in ${TEST_DIR}...${NC}"
    
    # Run the tests
    if [ "$VERBOSE" = true ]; then
        (cd "$TEST_DIR" && ctest --output-on-failure -V)
    else
        (cd "$TEST_DIR" && ctest --output-on-failure)
    fi
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}All tests passed successfully!${NC}"
    else
        echo -e "${RED}Some tests failed with exit code $exit_code${NC}"
        exit $exit_code
    fi
}

# Clean build artifacts
clean_build() {
    echo -e "${YELLOW}Cleaning build directories...${NC}"
    rm -rf "${MAIN_BUILD_DIR}"
    rm -rf "${TESTS_BUILD_DIR}"
    
    # Always clean the generated library files
    echo -e "${YELLOW}Cleaning library files...${NC}"
    
    # Remove shared library if it exists and there's an original backup
    if [ -f "${SCRIPT_DIR}/libxdnn.so" ]; then
        if [ -f "${SCRIPT_DIR}/libxdnn.so.orig" ]; then
            echo "Removing ${SCRIPT_DIR}/libxdnn.so"
            rm -f "${SCRIPT_DIR}/libxdnn.so"
        elif [ "$DEEP_CLEAN" = true ]; then
            # If no backup exists but deep clean is requested, delete anyway
            echo "Deep clean: removing ${SCRIPT_DIR}/libxdnn.so"
            rm -f "${SCRIPT_DIR}/libxdnn.so"
        else
            echo "Warning: ${SCRIPT_DIR}/libxdnn.so exists without backup file."
            echo "Use --deep option to remove all library files."
        fi
    fi
    
    # Remove static library if it exists and there's an original backup
    if [ -f "${SCRIPT_DIR}/libxdnn_static.a" ]; then
        if [ -f "${SCRIPT_DIR}/libxdnn_static.a.orig" ]; then
            echo "Removing ${SCRIPT_DIR}/libxdnn_static.a"
            rm -f "${SCRIPT_DIR}/libxdnn_static.a"
        elif [ "$DEEP_CLEAN" = true ]; then
            # If no backup exists but deep clean is requested, delete anyway
            echo "Deep clean: removing ${SCRIPT_DIR}/libxdnn_static.a"
            rm -f "${SCRIPT_DIR}/libxdnn_static.a"
        else
            echo "Warning: ${SCRIPT_DIR}/libxdnn_static.a exists without backup file."
            echo "Use --deep option to remove all library files."
        fi
    fi
    
    echo -e "${GREEN}Clean completed!${NC}"
}

# Build everything (libraries and tests)
build_all() {
    echo -e "${GREEN}Building libraries and tests...${NC}"
    
    # Skip checking for existing libraries
    CHECK_EXISTING_LIBS=false
    
    mkdir -p "${MAIN_BUILD_DIR}"
    cd "${MAIN_BUILD_DIR}"
    
    # Configure with CMake - WITH building the library
    cmake -DBUILD_XDNN_LIBRARY=ON ..
    
    # Build everything
    make -j${BUILD_JOBS}
    
    cd "${SCRIPT_DIR}"
    
    echo -e "${GREEN}All targets built successfully!${NC}"
}

# Execute the requested command
case "${COMMAND}" in
    lib)
        build_libraries
        ;;
    test)
        build_tests
        if [ "$RUN_TESTS" = true ]; then
            run_tests
        fi
        ;;
    all)
        build_all
        if [ "$RUN_TESTS" = true ]; then
            run_tests
        fi
        ;;
    clean)
        clean_build
        ;;
    help)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: ${COMMAND}${NC}"
        show_help
        exit 1
        ;;
esac

exit 0
