# Builds the CPython C extension `boidforge._native`.
#
# Raw CPython C-API + NumPy C-API only (no pybind11). The extension is installed
# into the importable `boidforge` package directory so it imports as
# `boidforge._native`.

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

python_add_library(_native MODULE
  src/boids_module.c
  src/boids_kernel.c
  WITH_SOABI)

target_include_directories(_native PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${Python_NumPy_INCLUDE_DIRS})

# Pin the NumPy C-API to a stable version and forbid deprecated symbols.
target_compile_definitions(_native PRIVATE
  NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

# Install next to the Python sources so `import boidforge._native` resolves.
install(TARGETS _native DESTINATION boidforge)
