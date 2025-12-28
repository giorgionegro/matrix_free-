cmake_minimum_required(VERSION 3.13)
project(matrix_free_benchmark LANGUAGES CXX)

find_package(deal.II REQUIRED HINTS ENV{DEAL_II_DIR})
deal_ii_initialize_cached_variables()



if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -march=native -mavx2 -mfma")
endif()

include_directories(include)

add_executable(benchmark_comprehensive_mpi src/benchmark_comprehensive_mpi.cpp)
deal_ii_setup_target(benchmark_comprehensive_mpi)

add_executable(benchmark_matrix_free_mpi src/benchmark_comprehensive_mpi.cpp)
target_compile_definitions(benchmark_matrix_free_mpi PRIVATE FORCE_ONLY_MF)
deal_ii_setup_target(benchmark_matrix_free_mpi)

add_executable(benchmark_matrix_based_mpi src/benchmark_comprehensive_mpi.cpp)
target_compile_definitions(benchmark_matrix_based_mpi PRIVATE FORCE_ONLY_MAT)
deal_ii_setup_target(benchmark_matrix_based_mpi)
