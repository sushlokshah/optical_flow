project(optical_flow)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 11)

include_directories(src/evaluation/cpp)
add_subdirectory(src)
