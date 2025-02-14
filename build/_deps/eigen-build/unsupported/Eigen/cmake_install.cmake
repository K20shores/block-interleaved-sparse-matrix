# Install script for directory: /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/Eigen/AdolcForward"
    "/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/Eigen/AlignedVector3"
    "/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/Eigen/ArpackSupport"
    "/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/Eigen/AutoDiff"
    "/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/Eigen/BVH"
    "/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/Eigen/EulerAngles"
    "/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/Eigen/FFT"
    "/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/Eigen/IterativeSolvers"
    "/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/Eigen/KroneckerProduct"
    "/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/Eigen/LevenbergMarquardt"
    "/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/Eigen/MatrixFunctions"
    "/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/Eigen/MoreVectorization"
    "/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/Eigen/MPRealSupport"
    "/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/Eigen/NonLinearOptimization"
    "/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/Eigen/NumericalDiff"
    "/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/Eigen/OpenGLSupport"
    "/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/Eigen/Polynomials"
    "/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/Eigen/Skyline"
    "/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/Eigen/SparseExtra"
    "/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/Eigen/SpecialFunctions"
    "/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/Eigen/Splines"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/unsupported/Eigen/CXX11/cmake_install.cmake")

endif()

