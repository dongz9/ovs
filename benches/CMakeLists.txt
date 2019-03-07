cmake_minimum_required(VERSION 2.8.12)
project(benches)

set(CMAKE_CXX_FLAGS "-std=c++1z -O3 -march=native ${CMAKE_CXX_FLAGS}")

add_executable(smc smc.cc)
add_executable(microbench microbench.cc)
