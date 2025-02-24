include(FetchContent)

################################################################################
# Eigen
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

################################################################################
# Cuda
if (BISM_ENABLE_CUDA)
  # Enable CUDA
  find_package(CUDA REQUIRED)
endif()

################################################################################
# MICM

if (BISM_ENABLE_MICM)
  FetchContent_Declare(micm
      GIT_REPOSITORY https://github.com/NCAR/micm.git
      GIT_TAG v.3.7.0
  )
  set(MICM_ENABLE_TESTS OFF)
  set(MICM_ENABLE_EXAMPLES OFF)

  FetchContent_MakeAvailable(micm)
endif()