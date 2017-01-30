cmake_minimum_required(VERSION 2.8)
project(wah C CXX)

set (CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# silence policy CMP0042 warning by enabling RPATH explicitly
if(APPLE AND NOT DEFINED CMAKE_MACOSX_RPATH)
  set(CMAKE_MACOSX_RPATH true)
endif()

# prohibit in-source builds
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds are not allowed. Please use "
                        "./configure to choose a build directory and "
                        "initialize the build configuration.")
endif ()

# check if the user provided CXXFLAGS on the command line
if (CMAKE_CXX_FLAGS)
  set(CXXFLAGS_PROVIDED true)
else()
  set(CMAKE_CXX_FLAGS "-std=c++14 -Wextra -Wall -pedantic -O3")
endif ()

# set build type (evaluate ENABLE_DEBUG flag)
if (ENABLE_DEBUG)
  set(CMAKE_BUILD_TYPE Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCPPA_DEBUG")
else ()
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif ()

# set(CAF_ROOT_DIR "/Users/noir/Git/actor-framework/build")
find_package(CAF COMPONENTS core io opencl)
if (CAF_FOUND)
  include_directories(${CAF_INCLUDE_DIRS})
else ()
  message(FATAL_ERROR "Could not find CAF")
endif ()

find_package(VAST)
if (VAST_FOUND)
  include_directories(${VAST_INCLUDE_DIRS})
else ()
  message(FATAL_ERROR "Could not find VAST")
endif ()

find_package(OpenCL REQUIRED)

include_directories(. include)

set(SOURCES_CPU
  src/cpu.cpp
)
set(SOURCES_GPU
  src/gpu.cpp
)
set(SOURCES_VAST
  src/vst.cpp
)
set(SOURCE_GENERATOR
  src/generate_test_data.cpp
)
file(GLOB_RECURSE HEADERS "include/*.hpp" "include/*.cl")

add_executable(cpu
  ${SOURCES_CPU}
  ${HEADERS}
)
target_link_libraries(cpu
  ${CMAKE_DL_LIBS}
  ${CAF_LIBRARY_CORE}
  ${CAF_LIBRARY_IO}
)

add_executable(gpu
  ${SOURCES_GPU}
  ${HEADERS}
  ${OpenCL_INCLUDE_DIRS}
  ${CAF_INCLUDE_DIR_OPENCL}
)
target_link_libraries(gpu
  ${CMAKE_DL_LIBS}
  ${CAF_LIBRARY_CORE}
  ${CAF_LIBRARY_IO}
  ${CAF_LIBRARY_OPENCL}
  ${OpenCL_LIBRARIES}
)

add_executable(vst
  ${SOURCES_VAST}
  ${HEADERS}
  ${VAST_INCLUDE_DIR}
)
target_link_libraries(vst
  ${CMAKE_DL_LIBS}
  ${CAF_LIBRARY_CORE}
  ${CAF_LIBRARY_IO}
  ${VAST_LIBRARY}
)

add_executable(generator
  ${SOURCE_GENERATOR}
  ${HEADERS}
)
target_link_libraries(generator
  ${CMAKE_DL_LIBS}
  ${CAF_LIBRARY_CORE}
)
