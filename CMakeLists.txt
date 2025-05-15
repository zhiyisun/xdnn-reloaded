cmake_minimum_required(VERSION 3.10)
project(xdnn VERSION 1.0.0 LANGUAGES CXX)

# Option to build the library (default: ON)
option(BUILD_XDNN_LIBRARY "Build the XDNN library" ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add compiler flags for optimization and architecture-specific features
add_compile_options(-mavx2 -mfma)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-march=native -O3)
endif()

# Define source files for the library
set(XDNN_SOURCES
    amx_sgemm_bf16bf16bf16.cpp
    amx_sgemm_bf16f8bf16.cpp
    amx_sgemm_f16f16f16.cpp
    bgemm_bf16bf16bf16.cpp
    bgemm_f32bf16f32.cpp
    hgemm_f16f16f32.cpp
    hgemm_f32f16f16.cpp
    hgemm_f32f16f32.cpp
    hgemm_f32s8f32.cpp
    hgemm_f32u4f32.cpp
    hgemm.cpp
    intrinsic_ext.cpp
    platform_detection.cpp
    sgemm_bf16bf16f32.cpp
    sgemm_f32bf16bf16.cpp
    sgemm_f32f16bf16.cpp
    sgemm_f32f16f32.cpp
    sgemm_f32nf4f32.cpp
    sgemm_f32s8f32.cpp
    sgemm_f32u4f32.cpp
    sgemm.cpp
    softmax.cpp
    transpose.cpp
    xdnn.cpp
)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(BUILD_XDNN_LIBRARY)
    # Build static library
    add_library(xdnn_static STATIC ${XDNN_SOURCES})
    set_target_properties(xdnn_static PROPERTIES 
        OUTPUT_NAME xdnn_static
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Build shared library
    add_library(xdnn SHARED ${XDNN_SOURCES})
    set_target_properties(xdnn PROPERTIES 
        OUTPUT_NAME xdnn
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Install targets
    install(TARGETS xdnn xdnn_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )

    # Install headers
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION include/xdnn
        FILES_MATCHING PATTERN "*.h"
    )
else()
    # Check if the libraries already exist
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libxdnn.so" OR NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libxdnn_static.a")
        message(FATAL_ERROR "XDNN libraries not found. Please build the libraries first or set BUILD_XDNN_LIBRARY=ON")
    endif()
    
    # Create imported targets for the pre-existing libraries
    add_library(xdnn SHARED IMPORTED GLOBAL)
    set_target_properties(xdnn PROPERTIES 
        IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/libxdnn.so"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    
    add_library(xdnn_static STATIC IMPORTED GLOBAL)
    set_target_properties(xdnn_static PROPERTIES
        IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/libxdnn_static.a"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    
    message(STATUS "Using pre-built XDNN libraries from: ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# Add the tests subdirectory
add_subdirectory(tests)
