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

# Historical H0 entry points.
option(VECTORFORGE_BUILD_RESIDENT_RETRIEVAL "Build the H0 resident retrieval experiment" OFF)
option(VECTORFORGE_BUILD_RESIDENT_PLACEMENT "Build the H0 placement experiment and private CUDA module" OFF)
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

# Each V2 executable can be enabled on its own. VECTORFORGE_V2_LOCALITY
# exposes the shared CUDA event diagnostics within these targets.
option(VECTORFORGE_BUILD_LOCALITY_STUDY "Build V2 locality and warmed replay" OFF)
option(VECTORFORGE_BUILD_PREPARED_STUDY "Build the V2 fixed-candidate preparation and lifecycle experiment" OFF)
option(VECTORFORGE_BUILD_MIXED_RETRIEVAL "Build the V2 mixed-candidate resident retrieval experiment" OFF)
if(VECTORFORGE_BUILD_LOCALITY_STUDY)
    add_executable(vectorforge_cuda_locality locality.cpp scoring.cu)
    target_compile_definitions(vectorforge_cuda_locality PRIVATE VECTORFORGE_V2_LOCALITY)
    list(APPEND cuda_consumers vectorforge_cuda_locality)
endif()
if(VECTORFORGE_BUILD_PREPARED_STUDY)
    add_executable(vectorforge_cuda_prepared prepared.cpp scoring.cu)
    target_compile_definitions(vectorforge_cuda_prepared PRIVATE VECTORFORGE_V2_LOCALITY VECTORFORGE_V2_PREPARED)
    list(APPEND cuda_consumers vectorforge_cuda_prepared)
endif()
if(VECTORFORGE_BUILD_MIXED_RETRIEVAL)
    add_executable(vectorforge_cuda_mixed_retrieval mixed_retrieval.cpp scoring.cu)
    target_compile_definitions(vectorforge_cuda_mixed_retrieval PRIVATE VECTORFORGE_V2_LOCALITY VECTORFORGE_V2_MIXED)
    list(APPEND cuda_consumers vectorforge_cuda_mixed_retrieval)
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
