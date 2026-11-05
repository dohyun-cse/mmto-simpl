cmake_minimum_required(VERSION 3.11)
cmake_policy(SET CMP0079 NEW) # Allow linking to CoreLib from subdirectories
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(mmto-simpl)
find_package(MPI REQUIRED)

# 1. Build type (defaults to Release)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()
message(STATUS "CMAKE_BUILD_TYPE is set to ${CMAKE_BUILD_TYPE}")

# Interface target collecting the headers, the library and MFEM
add_library(CoreLib INTERFACE)
target_include_directories(CoreLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

#--------------------------------------------------------------
# 2. MFEM configuration
# MFEM must be built with CMake and MPI. By default, MFEM is assumed to be in
# ../mfem/build (Release) and ../mfem/build-debug (Debug). Other locations can
# be given in a file user.cmake in this directory:
#   set(MFEM_RELEASE_DIR "/path/to/mfem/build")
#   set(MFEM_DEBUG_DIR "/path/to/mfem/build-debug")
if(EXISTS "${CMAKE_SOURCE_DIR}/user.cmake")
  include("${CMAKE_SOURCE_DIR}/user.cmake")
else()
  set(MFEM_RELEASE_DIR "${CMAKE_SOURCE_DIR}/../mfem/build")
  set(MFEM_DEBUG_DIR "${CMAKE_SOURCE_DIR}/../mfem/build-debug")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type)
if (_build_type STREQUAL "DEBUG")
  set(MFEM_DIR "${MFEM_DEBUG_DIR}")
elseif (_build_type STREQUAL "RELEASE")
  set(MFEM_DIR "${MFEM_RELEASE_DIR}")
else()
  message(FATAL_ERROR "Unknown build type: ${CMAKE_BUILD_TYPE}. Use Debug or Release.")
endif()
message(STATUS "MFEM_DIR is set to ${MFEM_DIR}")
include("${MFEM_DIR}/MFEMConfig.cmake")
target_link_libraries(CoreLib INTERFACE mfem)

#--------------------------------------------------------------
# 3. Library (lib/) and executables (src/)
add_subdirectory(lib)
add_subdirectory(src)
