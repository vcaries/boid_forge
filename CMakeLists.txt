cmake_minimum_required(VERSION 3.18)
project(boidforge LANGUAGES C)

# -----------------------------------------------------------------------------
# BoidForge native build.
#
# Builds the CPython C extension `boidforge._native` from native/. The extension
# uses the raw CPython C-API (no pybind11) and operates on contiguous, NumPy
# managed Struct-of-Arrays buffers. No simulation logic lives outside native/;
# no visualization logic lives anywhere in this target.
# -----------------------------------------------------------------------------

option(BOIDFORGE_BUILD_EXTENSION "Build the CPython C extension" ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Aggressive, portable optimization flags for the hot kernel.
if(MSVC)
  add_compile_options($<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/fp:fast>)
else()
  add_compile_options(
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-ffast-math>
    $<$<CONFIG:Release>:-funroll-loops>
    -Wall -Wextra)
endif()

if(BOIDFORGE_BUILD_EXTENSION)
  add_subdirectory(native)
endif()
