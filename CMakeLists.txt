cmake_minimum_required(VERSION 3.13)
project(matrix_test LANGUAGES CXX)

find_package(deal.II REQUIRED HINTS ENV{DEAL_II_DIR})

deal_ii_initialize_cached_variables()

# Enable vectorization flags for performance
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -flto")

  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math")

   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-vec-optimized")
endif()

include_directories(include)

# Main convergence test program
add_executable(matrix_test src/main.cpp)
deal_ii_setup_target(matrix_test)

# Matrix-free ADR solver
add_executable(matrix_free_solver src/main_matrix_free.cpp)
deal_ii_setup_target(matrix_free_solver)


