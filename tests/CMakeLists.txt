cmake_minimum_required(VERSION 3.10)
project(xdnn_tests)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check if we're running as a standalone project or as part of the main build
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # We are being built as a standalone project
    set(STANDALONE_BUILD TRUE)
else()
    # We are being built as part of the main project
    set(STANDALONE_BUILD FALSE)
endif()

# Find Google Test package if available
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    message(STATUS "GTest not found, downloading and building it")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Include directories
if(STANDALONE_BUILD)
    # When building tests only, include the parent directory
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
else()
    # When building as part of the main project
    include_directories(${CMAKE_SOURCE_DIR})
endif()

# Library directories - where to find libxdnn.so and libxdnn_static.a
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

# Add compiler flags
add_compile_options(-mavx2 -mfma)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-march=native)
endif()

# Add test executables - include platform detection in each test that needs it
add_executable(run_all_tests ${CMAKE_CURRENT_SOURCE_DIR}/run_all_tests.cpp ${PLATFORM_DETECTION_SRC})
add_executable(test_softmax ${CMAKE_CURRENT_SOURCE_DIR}/test_softmax.cpp ${PLATFORM_DETECTION_SRC})
add_executable(test_hgemm ${CMAKE_CURRENT_SOURCE_DIR}/test_hgemm.cpp ${PLATFORM_DETECTION_SRC})
add_executable(test_hgemm_f16f16f32 ${CMAKE_CURRENT_SOURCE_DIR}/test_hgemm_f16f16f32.cpp ${PLATFORM_DETECTION_SRC})
add_executable(test_hgemm_f32f16f32 ${CMAKE_CURRENT_SOURCE_DIR}/test_hgemm_f32f16f32.cpp ${PLATFORM_DETECTION_SRC})
add_executable(test_bgemm_f32bf16f32 ${CMAKE_CURRENT_SOURCE_DIR}/test_bgemm_f32bf16f32.cpp ${PLATFORM_DETECTION_SRC})
add_executable(test_amx_sgemm_bf16bf16bf16 ${CMAKE_CURRENT_SOURCE_DIR}/test_amx_sgemm_bf16bf16bf16.cpp ${PLATFORM_DETECTION_SRC})

# Link against Google Test and the XDNN library
if(STANDALONE_BUILD)
    # When building tests in standalone mode, need to first find the XDNN libraries
    find_library(XDNN_SHARED_LIB NAMES xdnn PATHS ${CMAKE_CURRENT_SOURCE_DIR}/..)
    find_library(XDNN_STATIC_LIB NAMES xdnn_static PATHS ${CMAKE_CURRENT_SOURCE_DIR}/..)
    
    if(NOT XDNN_SHARED_LIB)
        message(FATAL_ERROR "Could not find libxdnn.so in ${CMAKE_CURRENT_SOURCE_DIR}/..")
    endif()
    
    # Create imported targets for the XDNN libraries
    add_library(xdnn SHARED IMPORTED)
    set_target_properties(xdnn PROPERTIES 
        IMPORTED_LOCATION "${XDNN_SHARED_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
endif()

# Link the test executables - this code is the same regardless of build mode
target_link_libraries(run_all_tests GTest::gtest GTest::gtest_main xdnn)
target_link_libraries(test_softmax GTest::gtest GTest::gtest_main xdnn)
target_link_libraries(test_hgemm GTest::gtest GTest::gtest_main xdnn)
target_link_libraries(test_hgemm_f16f16f32 GTest::gtest GTest::gtest_main xdnn)
target_link_libraries(test_hgemm_f32f16f32 GTest::gtest GTest::gtest_main xdnn)
target_link_libraries(test_bgemm_f32bf16f32 GTest::gtest GTest::gtest_main xdnn)
target_link_libraries(test_amx_sgemm_bf16bf16bf16 GTest::gtest GTest::gtest_main xdnn)

# Add tests to CTest
enable_testing()
gtest_discover_tests(run_all_tests)
gtest_discover_tests(test_softmax)
gtest_discover_tests(test_hgemm)
gtest_discover_tests(test_hgemm_f16f16f32)
gtest_discover_tests(test_hgemm_f32f16f32)
gtest_discover_tests(test_bgemm_f32bf16f32)
gtest_discover_tests(test_amx_sgemm_bf16bf16bf16)

