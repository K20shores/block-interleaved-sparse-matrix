################################################################################
# Preamble
cmake_minimum_required(VERSION 3.21)

project(BlockInterleavedSparseMatrix LANGUAGES CXX VERSION 0.0.0)

################################################################################
# Projet wide setup options

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}/cmake")

option(BISM_ENABLE_CUDA "Enable cuda support" OFF)

################################################################################
# Dependencies

include(cmake/dependencies.cmake)

################################################################################
# the library

add_subdirectory(src)
