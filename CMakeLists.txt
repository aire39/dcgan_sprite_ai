cmake_minimum_required (VERSION 3.23.0)
project (spriteai)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 11)

IF (EXISTS ${PROJECT_SOURCE_DIR}/libs/pytorch_1_13_0_release AND EXISTS ${PROJECT_SOURCE_DIR}/libs/pytorch_1_13_0_debug)
  message("-- PYTORCH: Found pytorch library folders!")
else()
  message("-- PYTORCH: Pytorch folders missing. Will download them and extract them to libs folder")
  execute_process(
    COMMAND sh "${PROJECT_SOURCE_DIR}/scripts/libs/getlibs.sh"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/scripts/libs/"
)
endif()

set(PYTORCH_CAPI_RELEASE ${PROJECT_SOURCE_DIR}/libs/pytorch_1_13_0_release/libtorch/include/torch/csrc)
set(PYTORCH_INCLUDE_RELEASE ${PROJECT_SOURCE_DIR}/libs/pytorch_1_13_0_release/libtorch/include/torch/csrc/api/include)
set(PYTORCH_LIBS_RELEASE ${PROJECT_SOURCE_DIR}/libs/pytorch_1_13_0_release/libtorch/lib/)

set(PYTORCH_INCLUDE_TOP_DEBUG ${PROJECT_SOURCE_DIR}/libs/pytorch_1_13_0_debug/libtorch/include)
set(PYTORCH_INCLUDE_CAPI_DEBUG ${PROJECT_SOURCE_DIR}/libs/pytorch_1_13_0_debug/libtorch/include/torch/csrc)
set(PYTORCH_INCLUDE_DEBUG ${PROJECT_SOURCE_DIR}/libs/pytorch_1_13_0_debug/libtorch/include/torch/csrc/api/include)
set(PYTORCH_LIBS_DEBUG ${PROJECT_SOURCE_DIR}/libs/pytorch_1_13_0_debug/libtorch/lib/)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PYTORCH_INCLUDE_TOP_DEBUG})
include_directories(${PYTORCH_INCLUDE_CAPI_DEBUG})
include_directories(${PYTORCH_INCLUDE_DEBUG})

add_executable(spriteai
  capp/main.cpp
)

target_include_directories(spriteai PRIVATE
  "${PROJECT_SOURCE_DIR}/libs/pytorch_1_13_0_debug/libtorch/include"
)

target_link_libraries(spriteai
  ${PYTORCH_LIBS_DEBUG}/*.lib
)

#[[
execute_process(
  COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_SOURCE_DIR}/capp/build"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_SOURCE_DIR}/scripts/build_app/build.sh" "${PROJECT_SOURCE_DIR}/capp/build/"
)

#]]
