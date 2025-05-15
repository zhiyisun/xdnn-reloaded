cmake_minimum_required(VERSION 3.10)
project(xdnn_tests)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
include_directories(${CMAKE_SOURCE_DIR}/..)

# Add compiler flags
add_compile_options(-mavx2 -mfma)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-march=native)
endif()

# Add platform detection source
set(PLATFORM_DETECTION_SRC ${CMAKE_SOURCE_DIR}/../platform_detection.cpp)

# Add test executables
add_executable(test_sgemm test_sgemm.cpp ${PLATFORM_DETECTION_SRC})
add_executable(test_hgemm test_hgemm.cpp ${PLATFORM_DETECTION_SRC})
add_executable(test_bgemm test_bgemm.cpp ${PLATFORM_DETECTION_SRC})
add_executable(test_amx_sgemm test_amx_sgemm.cpp ${PLATFORM_DETECTION_SRC})
add_executable(test_softmax test_softmax.cpp ${PLATFORM_DETECTION_SRC})
add_executable(test_transpose test_transpose.cpp ${CMAKE_SOURCE_DIR}/../transpose_impl.cpp ${PLATFORM_DETECTION_SRC})
add_executable(test_data_types test_data_types.cpp ${PLATFORM_DETECTION_SRC})
add_executable(test_basics test_basics.cpp)
add_executable(run_all_tests run_all_tests.cpp ${PLATFORM_DETECTION_SRC})

# Link against Google Test and the XDNN library
target_link_libraries(test_sgemm GTest::gtest GTest::gtest_main ${CMAKE_SOURCE_DIR}/../libxdnn.so)
target_link_libraries(test_hgemm GTest::gtest GTest::gtest_main ${CMAKE_SOURCE_DIR}/../libxdnn.so)
target_link_libraries(test_bgemm GTest::gtest GTest::gtest_main ${CMAKE_SOURCE_DIR}/../libxdnn.so)
target_link_libraries(test_amx_sgemm GTest::gtest GTest::gtest_main ${CMAKE_SOURCE_DIR}/../libxdnn.so)
target_link_libraries(test_softmax GTest::gtest GTest::gtest_main ${CMAKE_SOURCE_DIR}/../libxdnn.so)
target_link_libraries(test_transpose GTest::gtest GTest::gtest_main ${CMAKE_SOURCE_DIR}/../libxdnn.so)
target_link_libraries(test_data_types GTest::gtest GTest::gtest_main ${CMAKE_SOURCE_DIR}/../libxdnn.so)
target_link_libraries(test_basics GTest::gtest GTest::gtest_main)
target_link_libraries(run_all_tests GTest::gtest GTest::gtest_main ${CMAKE_SOURCE_DIR}/../libxdnn.so)

# Add tests to CTest
enable_testing()
add_test(NAME test_basics COMMAND test_basics)
add_test(NAME test_transpose COMMAND test_transpose)
add_test(NAME test_data_types COMMAND test_data_types)
# These tests are disabled until we implement the actual functionality
# add_test(NAME test_sgemm COMMAND test_sgemm)
# add_test(NAME test_hgemm COMMAND test_hgemm)
# add_test(NAME test_bgemm COMMAND test_bgemm)
# add_test(NAME test_amx_sgemm COMMAND test_amx_sgemm)
# add_test(NAME test_softmax COMMAND test_softmax)
gtest_discover_tests(test_sgemm)
gtest_discover_tests(test_hgemm)
gtest_discover_tests(test_bgemm)
gtest_discover_tests(test_amx_sgemm)
gtest_discover_tests(test_softmax)
gtest_discover_tests(test_transpose)
gtest_discover_tests(test_data_types)
gtest_discover_tests(run_all_tests)
