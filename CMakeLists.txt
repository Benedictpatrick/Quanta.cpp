cmake_minimum_required(VERSION 3.20)
project(quanta CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(quanta_engine STATIC
  engine/src/model_file.cpp
  engine/src/model.cpp
  engine/src/ops_ref.cpp
  engine/src/tokenizer.cpp
  engine/src/sampler.cpp
  engine/src/threadpool.cpp
  engine/src/ops_dispatch.cpp
  engine/src/session.cpp
  engine/src/checks.cpp
)
target_include_directories(quanta_engine PUBLIC engine/src engine/include)
target_compile_options(quanta_engine PRIVATE -Wall -Wextra)

# Host builds: let the compiler use this machine's SIMD (auto-vectorizes the reference kernels).
option(QUANTA_NATIVE "Compile for the host CPU (-march=native)" ON)
if(QUANTA_NATIVE AND NOT CMAKE_CROSSCOMPILING)
  target_compile_options(quanta_engine PRIVATE -march=native)
endif()

# ARM64: NEON kernels. The dot-product variant is compiled for ARMv8.2+dotprod and only used when
# the CPU reports support at runtime; the base variant runs on every ARMv8 core (e.g. Cortex-A53).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  target_sources(quanta_engine PRIVATE engine/src/ops_neon_base.cpp engine/src/ops_neon_dot.cpp)
  set_source_files_properties(engine/src/ops_neon_dot.cpp PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod+fp16")
endif()

option(QUANTA_BUILD_CLI "Build the quanta command-line tool" ON)
if(QUANTA_BUILD_CLI)
  add_executable(quanta tools/cli/main.cpp)
  target_link_libraries(quanta PRIVATE quanta_engine)
  if(MINGW)
    target_link_options(quanta PRIVATE -static)
  endif()
endif()
