cmake_minimum_required (VERSION 3.23.0)
project (spriteai)

enable_language(CUDA)

if(${CMAKE_VERSION} VERSION_LESS "3.26.0")
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CUDA_STANDARD 17)
  message("cmake version less than 3.26 available. Will use -std17")
else()
  if (${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER "12.1.0")
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CUDA_STANDARD 20)
    message("Cuda 12.1 and cmake 3.26 available. Will use -std20")
  else()
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CUDA_STANDARD 17)
    message("Cuda less than 12.1 or cmake version less than 3.26 available. Will use -std17")
  endif()
endif()

set(CMAKE_C_STANDARD 11)

message("Using CUDA Version: ${CMAKE_CUDA_COMPILER_VERSION}")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Build Debug")
  list(APPEND CMAKE_PREFIX_PATH "libs/pytorch_1_13_0_debug/libtorch/")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  message("Build Release")
  list(APPEND CMAKE_PREFIX_PATH "libs/pytorch_1_13_0_release/libtorch/")
else()
  message("Build Debug")
  list(APPEND CMAKE_PREFIX_PATH "libs/pytorch_1_13_0_debug/libtorch/")
endif()

find_package(Torch REQUIRED)

IF (EXISTS ${PROJECT_SOURCE_DIR}/libs/pytorch_1_13_0_release AND EXISTS ${PROJECT_SOURCE_DIR}/libs/pytorch_1_13_0_debug)
  message("-- PYTORCH: Found pytorch library folders!")
else()
  message("-- PYTORCH: Pytorch folders missing. Will download them and extract them to libs folder")
  execute_process(
    COMMAND sh "${PROJECT_SOURCE_DIR}/scripts/libs/getlibs.sh"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/scripts/libs/"
  )
endif()

add_executable(spriteai
  capp/main.cpp
)

message("PYTORCH LIBS: ${TORCH_LIBRARIES}")

target_link_libraries(spriteai
  ${TORCH_LIBRARIES}
)
