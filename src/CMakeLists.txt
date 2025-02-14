add_library(bism)
add_library(ncar::bism ALIAS bism)

target_compile_features(bism PUBLIC cxx_std_20)

target_sources(bism
  PRIVATE
    empty.cpp
)

target_include_directories(bism
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(bism 
  PUBLIC 
    Eigen3::Eigen 
)

if (BISM_ENABLE_CUDA)
  target_link_libraries(bism 
    PUBLIC 
      CUDA::cusparse
  )
endif()

add_executable(BlockSparseLU main.cpp)
target_link_libraries(BlockSparseLU PRIVATE ncar::bism)