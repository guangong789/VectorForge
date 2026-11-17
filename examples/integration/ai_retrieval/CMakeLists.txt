cmake_minimum_required(VERSION 3.20)
project(VectorForgeAIRetrievalConsumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(VECTORFORGE_SOURCE_DIR "" CACHE PATH "VectorForge source checkout")
if(NOT EXISTS "${VECTORFORGE_SOURCE_DIR}/include/vectorforge/validated_dataset.hpp")
    message(FATAL_ERROR "Set VECTORFORGE_SOURCE_DIR to a VectorForge source checkout")
endif()

# This standalone project tests public consumption; the library suite has its own build.
set(BUILD_TESTING OFF CACHE BOOL "Disable the embedded library's tests" FORCE)
foreach(option_name IN ITEMS VECTORFORGE_BUILD_BENCHMARKS VECTORFORGE_SIMILARITY_DIAGNOSTICS
        VECTORFORGE_BUILD_FMA_COMPARISON VECTORFORGE_BUILD_ACCUMULATOR_STUDY
        VECTORFORGE_BUILD_CROSS_ROW_STUDY VECTORFORGE_BUILD_AVX2_SCORING VECTORFORGE_TEST_SIMD_DISPATCH)
    set(${option_name} OFF CACHE BOOL "Disabled for the independent consumer" FORCE)
endforeach()
add_subdirectory("${VECTORFORGE_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/vectorforge")

add_executable(vectorforge_ai_retrieval main.cpp)
target_link_libraries(vectorforge_ai_retrieval PRIVATE VectorForge::vectorforge)
if(MSVC)
    target_compile_options(vectorforge_ai_retrieval PRIVATE /W4 /permissive-)
else()
    target_compile_options(vectorforge_ai_retrieval PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
endif()

# Independent exit-status test, even though the embedded library's BUILD_TESTING is OFF.
enable_testing()
add_test(NAME ai_retrieval_consumer COMMAND vectorforge_ai_retrieval)
