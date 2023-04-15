cmake_minimum_required (VERSION 3.23.0)
project (dcgan_example)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

find_package(Torch REQUIRED)

add_executable(dcgan_example
  main.cpp
)

target_link_libraries(dcgan_example
  ${TORCH_LIBRARIES}
)

target_include_directories (dcgan_example PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR})

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  FILE(GLOB libtorch_dllfiles "${LIBTORCH_PATH_DEBUG}/lib/*.dll")
  add_custom_command(TARGET dcgan_example POST_BUILD
          WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
          COMMAND ${CMAKE_COMMAND} -E copy ${libtorch_dllfiles} ${CMAKE_CURRENT_BINARY_DIR})

  FILE(GLOB libtorch_pdbfiles "${LIBTORCH_PATH_DEBUG}/lib/*.pdb")
  add_custom_command(TARGET dcgan_example POST_BUILD
          WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
          COMMAND ${CMAKE_COMMAND} -E copy ${libtorch_pdbfiles} ${CMAKE_CURRENT_BINARY_DIR})

elseif(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  FILE(GLOB libtorch_dllfiles "${LIBTORCH_PATH_RELEASE}/lib/*.dll")
  add_custom_command(TARGET dcgan_example POST_BUILD
          WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
          COMMAND ${CMAKE_COMMAND} -E copy ${libtorch_dllfiles} ${CMAKE_CURRENT_BINARY_DIR})
endif ()