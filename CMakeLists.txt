cmake_minimum_required(VERSION 3.18)
project(fp4-attention LANGUAGES CUDA)

set(CMAKE_CUDA_ARCHITECTURES 120a)

add_executable(test_fp4 tests/test_fp4.cu)
target_include_directories(test_fp4 PRIVATE ${CMAKE_SOURCE_DIR})
