cmake_minimum_required(VERSION 3.20)
project(quanta_android CXX)

# The engine itself comes from the repository root.
set(QUANTA_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(QUANTA_NATIVE OFF CACHE BOOL "" FORCE)
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../.. quanta_engine_build)

add_library(quanta_jni SHARED quanta_jni.cpp)
target_compile_options(quanta_jni PRIVATE -O2 -Wall)
target_link_libraries(quanta_jni PRIVATE quanta_engine log)
