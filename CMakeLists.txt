cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)
 
include(CTest)

include_directories(${CMAKE_SOURCE_DIR}/src)

project(cuckoo_win_bundle)
add_library(cuckoo_win_bundle STATIC
src/crypto/blake2b-ref.c
src/windows/getopt.c
)
 
target_compile_features(cuckoo_win_bundle PUBLIC cxx_std_11)
 
set_target_properties(cuckoo_win_bundle PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
add_executable(mean_gpu src/cuckoo/mean.cu)
 
set_property(TARGET mean_gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
			 
target_link_libraries(mean_gpu PRIVATE cuckoo_win_bundle)
 







