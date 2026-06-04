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

# Optimization flags for the hot kernel.
#
# IMPORTANT: the kernel must stay bit-identical to the NumPy backends (CLAUDE.md
# §3), which reproduces NumPy's float32 pairwise summation exactly. That forbids
# fast-math: no reassociation and no FMA contraction, or the carefully ordered
# sums would re-round. We therefore use *precise* floating point and still keep
# -O2/-O3 and loop unrolling (which do not change FP results).
if(MSVC)
  add_compile_options(
    $<$<CONFIG:Release>:/O2>
    $<$<CONFIG:Release>:/fp:precise>)
else()
  add_compile_options(
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-fno-fast-math>
    $<$<CONFIG:Release>:-ffp-contract=off>
    $<$<CONFIG:Release>:-funroll-loops>
    -Wall -Wextra)
endif()

if(BOIDFORGE_BUILD_EXTENSION)
  add_subdirectory(native)
endif()
