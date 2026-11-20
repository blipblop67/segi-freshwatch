cmake_minimum_required(VERSION 3.18)
project(freshcuda LANGUAGES CXX CUDA)

# Orin = sm_87. Override with -DCMAKE_CUDA_ARCHITECTURES=110 for Thor.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_library(freshcuda SHARED src/freshcuda.cu)
target_include_directories(freshcuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(freshcuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>)
set_target_properties(freshcuda PROPERTIES
  CUDA_RUNTIME_LIBRARY Shared)

install(TARGETS freshcuda LIBRARY DESTINATION lib)
install(FILES include/freshcuda.h DESTINATION include)
