cmake_minimum_required(VERSION 3.16)

project(attention_primitives VERSION 0.1.0 LANGUAGES CXX)

add_library(attention_primitives INTERFACE)
add_library(attention_primitives::attention_primitives ALIAS attention_primitives)
target_compile_features(attention_primitives INTERFACE cxx_std_17)
target_include_directories(
    attention_primitives
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

option(ATTENTION_BUILD_EXAMPLES "Build the six operator examples" ON)
if(ATTENTION_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    add_test(
        NAME test_sparse_cpu
        COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_cpu_tests.py sparse
    )
    add_test(
        NAME test_linear_cpu
        COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_cpu_tests.py linear
    )
    set_tests_properties(
        test_sparse_cpu test_linear_cpu
        PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
    )
endif()
