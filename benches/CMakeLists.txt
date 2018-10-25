cmake_minimum_required(VERSION 2.8.12)
project(benches)

set(CMAKE_CXX_FLAGS "-std=c++1z ${CMAKE_CXX_FLAGS}")

add_subdirectory(abseil-cpp)

add_executable(smc smc.cc)
target_include_directories(smc PRIVATE abseil-cpp)
target_link_libraries(smc absl::base absl::container absl::hash)
