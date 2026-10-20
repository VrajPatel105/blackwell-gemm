cmake_minimum_required(VERSION 3.24)
project(cuda_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)          
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)        
endif()

find_package(CUDAToolkit REQUIRED)

file(GLOB KERNEL_SRCS CONFIGURE_DEPENDS src/kernels/*.cu)

add_executable(gemm
  src/bench/main.cu
  src/bench/matrix.cu
  src/bench/reference.cu
  src/bench/timing.cu
  ${KERNEL_SRCS})
target_include_directories(gemm PRIVATE include)
target_compile_options(gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_link_libraries(gemm PRIVATE CUDA::cublas)

add_executable(ceilings src/bench/ceilings.cu)
target_include_directories(ceilings PRIVATE include)
target_link_libraries(ceilings PRIVATE CUDA::cublas)