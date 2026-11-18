cmake_minimum_required(VERSION 3.20)
project(VectorForgeCudaScoringPilot LANGUAGES CXX CUDA)

set(CUDA_OPERATOR_INCLUDE_DIR "" CACHE PATH "CudaOperator include directory")
if(NOT EXISTS "${CUDA_OPERATOR_INCLUDE_DIR}/cudaHelper/cuda_check.cuh")
    message(FATAL_ERROR "Set CUDA_OPERATOR_INCLUDE_DIR to the CudaOperator include directory")
endif()
find_package(CUDAToolkit 12.4 REQUIRED)

set(BUILD_TESTING OFF)
set(VECTORFORGE_BUILD_AVX2_SCORING OFF)
add_subdirectory("../../.." "${CMAKE_CURRENT_BINARY_DIR}/vectorforge")

option(VECTORFORGE_BUILD_RESIDENT_RETRIEVAL "Build the isolated resident retrieval evaluation" OFF)
option(VECTORFORGE_BUILD_RESIDENT_PLACEMENT "Build the private resident placement experiment" OFF)
add_executable(vectorforge_cuda_scoring_pilot main.cpp scoring.cu)
set(cuda_consumers vectorforge_cuda_scoring_pilot)
if(VECTORFORGE_BUILD_RESIDENT_RETRIEVAL)
    add_executable(vectorforge_cuda_resident_retrieval retrieval.cpp scoring.cu)
    list(APPEND cuda_consumers vectorforge_cuda_resident_retrieval)
endif()

if(VECTORFORGE_BUILD_RESIDENT_PLACEMENT)
    # Only the standalone build needs a PIC copy of the unchanged CPU library.
    set_target_properties(vectorforge PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(vectorforge_placement_cuda MODULE placement_cuda.cpp scoring.cu)
    set_target_properties(vectorforge_placement_cuda PROPERTIES CXX_VISIBILITY_PRESET hidden
        CUDA_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
    list(APPEND cuda_consumers vectorforge_placement_cuda)
    add_executable(vectorforge_resident_placement placement.cpp)
    target_link_libraries(vectorforge_resident_placement PRIVATE VectorForge::vectorforge ${CMAKE_DL_LIBS})
    set_target_properties(vectorforge_resident_placement PROPERTIES CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS OFF)
    target_compile_options(vectorforge_resident_placement PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -fno-fast-math)
    add_dependencies(vectorforge_resident_placement vectorforge_placement_cuda)
endif()

foreach(consumer IN LISTS cuda_consumers)
    target_link_libraries(${consumer} PRIVATE VectorForge::vectorforge CUDA::cudart)
    target_include_directories(${consumer} PRIVATE "${CUDA_OPERATOR_INCLUDE_DIR}")
    set_target_properties(${consumer} PROPERTIES
        CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS OFF
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES CUDA_EXTENSIONS OFF
        CUDA_ARCHITECTURES 86-real CUDA_RUNTIME_LIBRARY Shared)
    target_compile_options(${consumer} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wsign-conversion;-fno-fast-math>
        $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false;--fmad=false;-lineinfo;-Xcompiler=-Wall,-Wextra,-Wconversion,-Wsign-conversion>)
endforeach()
