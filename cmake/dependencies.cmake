include(FetchContent)

# Fetch Eigen
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

if (BISM_ENABLE_CUDA)
  # Enable CUDA
  find_package(CUDA REQUIRED)
endif()
