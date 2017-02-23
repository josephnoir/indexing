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

if(CAF_ENABLE_ADDRESS_SANITIZER)
  # check whether address sanitizer is available
  set(CXXFLAGS_BACKUP "${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
  try_run(ProgramResult
          CompilationSucceeded
          "${CMAKE_CURRENT_BINARY_DIR}"
          "${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_compiler_version.cpp")
  if(NOT CompilationSucceeded)
    message(WARNING "Address Sanitizer is not available on selected compiler")
  else()
    message(STATUS "Enable Address Sanitizer")
    set(EXTRA_FLAGS "${EXTRA_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
  endif()
  # restore CXX flags
  set(CMAKE_CXX_FLAGS "${CXXFLAGS_BACKUP}")
endif(CAF_ENABLE_ADDRESS_SANITIZER)

# check if the user provided CXXFLAGS, set defaults otherwise
if(NOT CMAKE_CXX_FLAGS)
  set(CMAKE_CXX_FLAGS                   "-std=c++14 -Wextra -Wall -pedantic ${EXTRA_FLAGS}")
endif()
if(NOT CMAKE_CXX_FLAGS_DEBUG)
  set(CMAKE_CXX_FLAGS_DEBUG             "-O0 -g")
endif()
if(NOT CMAKE_CXX_FLAGS_MINSIZEREL)
  set(CMAKE_CXX_FLAGS_MINSIZEREL        "-Os")
endif()
if(NOT CMAKE_CXX_FLAGS_RELEASE)
  set(CMAKE_CXX_FLAGS_RELEASE           "-O3 -DNDEBUG")
endif()
if(NOT CMAKE_CXX_FLAGS_RELWITHDEBINFO)
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO    "-O2 -g")
endif()

# set build default build type to RelWithDebInfo if not set
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" UPPER_BUILD_TYPE)
set(ALL_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${UPPER_BUILD_TYPE}}")

message(STATUS "CXX_FLAGS: ${ALL_CXX_FLAGS}")

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
  message(STATUS "Could not find VAST")
endif ()

find_package(OpenCL REQUIRED)

include_directories(. include)

set(SOURCES_CPU
  src/cpu.cpp
)
set(SOURCES_GPU
  src/gpu.cpp
)
set(SOURCES_PHASES
  src/phases.cpp
)
set(SOURCES_VAST
  src/vst.cpp
)
set(SOURCE_GENERATOR
  src/generate_test_data.cpp
)
set(SOURCE_PLAIN
  src/plain.cpp
)
set(SOURCES_OVERHEAD
  src/overhead_benchmark.cpp
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

add_executable(phases
  ${SOURCES_PHASES}
  ${HEADERS}
  ${OpenCL_INCLUDE_DIRS}
  ${CAF_INCLUDE_DIR_OPENCL}
)
target_link_libraries(phases
  ${CMAKE_DL_LIBS}
  ${CAF_LIBRARY_CORE}
  ${CAF_LIBRARY_IO}
  ${CAF_LIBRARY_OPENCL}
  ${OpenCL_LIBRARIES}
)

if (VAST_FOUND)
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
endif()

add_executable(generator
  ${SOURCE_GENERATOR}
  ${HEADERS}
)
target_link_libraries(generator
  ${CMAKE_DL_LIBS}
  ${CAF_LIBRARY_CORE}
)

add_executable(plain
  ${SOURCE_PLAIN}
  ${HEADERS}
  ${OpenCL_INCLUDE_DIRS}
)
target_link_libraries(plain
  ${CMAKE_DL_LIBS}
  ${OpenCL_LIBRARIES}
  ${CAF_LIBRARY_CORE}
  ${CAF_LIBRARY_IO}
)

add_executable(overhead
  ${SOURCES_OVERHEAD}
  ${HEADERS}
  ${OpenCL_INCLUDE_DIRS}
  ${CAF_INCLUDE_DIR_OPENCL}
)
target_link_libraries(overhead
  ${CMAKE_DL_LIBS}
  ${CAF_LIBRARY_CORE}
  ${CAF_LIBRARY_IO}
  ${CAF_LIBRARY_OPENCL}
  ${OpenCL_LIBRARIES}
)
