cmake_minimum_required (VERSION 3.23.0)
project (spriteai)

enable_language(CUDA)

set(LIBTORCH_PATH_DEBUG "${PROJECT_SOURCE_DIR}/libs/pytorch_1_13_0_debug/libtorch" CACHE FILEPATH "set libtorch debug folder")
set(LIBTORCH_PATH_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/libs/pytorch_1_13_0_release/libtorch" CACHE FILEPATH "set libtorch release folder")
set(BUILD_EXAMPLES FALSE CACHE BOOL "build examples")

message("CMAKE_VERSION: ${CMAKE_VERSION}")

if(${CMAKE_VERSION} VERSION_LESS "3.26.0" OR ${CMAKE_VERSION} VERSION_LESS "3.26.0-msvc1")
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CUDA_STANDARD 17)
  message("cmake version less than 3.26 available. Will use c++17")
else()
  if (${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER "12.1.0")
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CUDA_STANDARD 20)
    message("Cuda 12.1 and cmake 3.26 available. Will use c++20")
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
  list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH_DEBUG}")
  if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    message("Using MSVC")
  else ()
    message("Using GCC")
    add_compile_options(-O0 -g)
    add_definitions(/D_SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING)
  endif ()
elseif(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  message("Build Release")
  list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH_RELEASE}")
  if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    message("Using MSVC")
    add_compile_options(/Ot)
    add_definitions(/D_SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING)
  else()
    message("Using GCC")
    add_compile_options(-O3 -g)
  endif ()
else()
  message("Build Debug")
  list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH_DEBUG}")
endif()

IF (EXISTS ${PROJECT_SOURCE_DIR}/libs/pytorch_1_13_0_release AND EXISTS ${PROJECT_SOURCE_DIR}/libs/pytorch_1_13_0_debug)
  message("-- PYTORCH: Found pytorch library folders!")
else()
  message("-- PYTORCH: Pytorch folders missing. Will download them and extract them to libs folder")
  execute_process(
          COMMAND sh "${PROJECT_SOURCE_DIR}/scripts/libs/getlibs.sh"
          WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/scripts/libs"
  )
endif()

include_directories(libs)

set(BUILD_SHARED_LIBS FALSE) # build using the static libraries
add_subdirectory(libs/sfml)

find_package (Threads REQUIRED)
find_package(Torch REQUIRED)
message("PYTORCH LIBS: ${TORCH_LIBRARIES}")

if (${BUILD_EXAMPLES} )
add_subdirectory(cexamples/dcgan)
endif (${BUILD_EXAMPLES})

add_executable(spriteai
        capp/main.cpp
        capp/GeneratorImpl.cpp capp/GeneratorImpl.h capp/DiscriminatorImpl.cpp capp/ImageFolder.cpp capp/ImageFolder.h capp/SeqDiscriminator.cpp capp/SeqDiscriminator.h capp/DCGANUtils.h capp/DCGANUtils.cpp capp/Window.cpp capp/Window.h capp/DrawGraph.cpp capp/DrawGraph.h)
target_link_libraries(${CMAKE_PROJECT_NAME}
        ${TORCH_LIBRARIES}
        sfml-window
        sfml-graphics
        Threads::Threads
        )

target_include_directories(spriteai PUBLIC
        $<TARGET_PROPERTY:sfml-window,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:sfml-graphics,INTERFACE_INCLUDE_DIRECTORIES>)

if (CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  target_link_options(${CMAKE_PROJECT_NAME} PUBLIC /DEBUG:FULL)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  FILE(GLOB libtorch_dllfiles "${LIBTORCH_PATH_DEBUG}/lib/*.dll")
  add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
          WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
          COMMAND ${CMAKE_COMMAND} -E copy ${libtorch_dllfiles} ${CMAKE_CURRENT_BINARY_DIR})

  FILE(GLOB libtorch_pdbfiles "${LIBTORCH_PATH_DEBUG}/lib/*.pdb")
  add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
          WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
          COMMAND ${CMAKE_COMMAND} -E copy ${libtorch_pdbfiles} ${CMAKE_CURRENT_BINARY_DIR})

elseif(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  FILE(GLOB libtorch_dllfiles "${LIBTORCH_PATH_RELEASE}/lib/*.dll")
  add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
          WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
          COMMAND ${CMAKE_COMMAND} -E copy ${libtorch_dllfiles} ${CMAKE_CURRENT_BINARY_DIR})
endif ()
