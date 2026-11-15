cmake_minimum_required(VERSION 3.20)
project(VectorForge VERSION 0.1.0 LANGUAGES CXX)

option(VECTORFORGE_ENABLE_SANITIZERS "Enable AddressSanitizer and UBSan" OFF)
option(VECTORFORGE_BUILD_BENCHMARKS "Build the standalone benchmark and its smoke tests" OFF)
option(VECTORFORGE_SIMILARITY_DIAGNOSTICS "Save GCC reports and assembly for the benchmark-only similarity kernel" OFF)
option(VECTORFORGE_BUILD_FMA_COMPARISON "Build isolated GCC/Clang x86-64 scalar FMA target comparisons" OFF)

add_library(vectorforge
    src/column_batch.cpp
    src/embedding_buffer.cpp
    src/batch_view.cpp
    src/scalar_reference.cpp
    src/batched_executor.cpp
    src/validated_dataset.cpp
)
add_library(VectorForge::vectorforge ALIAS vectorforge)
target_include_directories(vectorforge PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(vectorforge PUBLIC cxx_std_20)
set_target_properties(vectorforge PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(vectorforge PRIVATE /W4 /permissive-)
else()
    target_compile_options(vectorforge PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
endif()

if(VECTORFORGE_ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$" AND NOT MSVC)
        target_compile_options(vectorforge PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(vectorforge PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    else()
        message(FATAL_ERROR "VECTORFORGE_ENABLE_SANITIZERS requires GCC or Clang with ASan/UBSan support")
    endif()
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(vectorforge_tests tests/data_foundation_test.cpp)
    target_link_libraries(vectorforge_tests PRIVATE VectorForge::vectorforge)
    set_target_properties(vectorforge_tests PROPERTIES CXX_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(vectorforge_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(vectorforge_tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
    endif()
    add_test(NAME data_foundation COMMAND vectorforge_tests)

    add_executable(vectorforge_scalar_tests tests/scalar_reference_test.cpp)
    target_link_libraries(vectorforge_scalar_tests PRIVATE VectorForge::vectorforge)
    set_target_properties(vectorforge_scalar_tests PROPERTIES CXX_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(vectorforge_scalar_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(vectorforge_scalar_tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
    endif()
    add_test(NAME scalar_reference COMMAND vectorforge_scalar_tests)

    add_executable(vectorforge_batched_tests tests/batched_executor_test.cpp)
    target_link_libraries(vectorforge_batched_tests PRIVATE VectorForge::vectorforge)
    set_target_properties(vectorforge_batched_tests PROPERTIES CXX_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(vectorforge_batched_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(vectorforge_batched_tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
    endif()
    add_test(NAME batched_executor COMMAND vectorforge_batched_tests)

    add_executable(vectorforge_dataset_tests tests/validated_dataset_test.cpp)
    target_link_libraries(vectorforge_dataset_tests PRIVATE VectorForge::vectorforge)
    set_target_properties(vectorforge_dataset_tests PROPERTIES CXX_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(vectorforge_dataset_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(vectorforge_dataset_tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
    endif()
    add_test(NAME validated_dataset COMMAND vectorforge_dataset_tests)
endif()

if(VECTORFORGE_BUILD_BENCHMARKS)
    add_library(vectorforge_similarity_kernel STATIC benchmarks/similarity_kernel.cpp)
    target_link_libraries(vectorforge_similarity_kernel PUBLIC VectorForge::vectorforge)
    set_target_properties(vectorforge_similarity_kernel PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
    if(VECTORFORGE_SIMILARITY_DIAGNOSTICS)
        if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            message(FATAL_ERROR "VECTORFORGE_SIMILARITY_DIAGNOSTICS requires GCC")
        endif()
        target_compile_options(vectorforge_similarity_kernel PRIVATE
            "-fopt-info-vec-all=${CMAKE_CURRENT_BINARY_DIR}/similarity-vectorization-$<CONFIG>.txt"
            -save-temps=obj)
    endif()
    add_library(vectorforge_benchmark_support STATIC benchmarks/benchmark.cpp benchmarks/topk.cpp benchmarks/reuse.cpp benchmarks/similarity.cpp)
    target_link_libraries(vectorforge_benchmark_support PUBLIC VectorForge::vectorforge)
    target_link_libraries(vectorforge_benchmark_support PUBLIC vectorforge_similarity_kernel)
    target_include_directories(vectorforge_benchmark_support PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks")
    target_include_directories(vectorforge_benchmark_support PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/benchmark-info/$<CONFIG>")
    # Record the active configuration/global flags and core target options, rather
    # than assuming that Release always means a particular optimization level.
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/benchmark-info/$<CONFIG>/benchmark_build_info.hpp" CONTENT
"#pragma once
namespace vectorforge::benchmark::build_info {
inline constexpr char compiler[] = R\"vf(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})vf\";
inline constexpr char build_type[] = R\"vf($<CONFIG>)vf\";
inline constexpr char optimization_flags[] = R\"vf(C++20; ${CMAKE_CXX_FLAGS} $<$<CONFIG:Debug>:${CMAKE_CXX_FLAGS_DEBUG}>$<$<CONFIG:Release>:${CMAKE_CXX_FLAGS_RELEASE}>$<$<CONFIG:RelWithDebInfo>:${CMAKE_CXX_FLAGS_RELWITHDEBINFO}>$<$<CONFIG:MinSizeRel>:${CMAKE_CXX_FLAGS_MINSIZEREL}> $<JOIN:$<TARGET_PROPERTY:vectorforge,COMPILE_OPTIONS>, >)vf\";
inline constexpr bool sanitized = $<BOOL:${VECTORFORGE_ENABLE_SANITIZERS}>;
}
")
    add_executable(vectorforge_benchmark benchmarks/main.cpp)
    target_link_libraries(vectorforge_benchmark PRIVATE vectorforge_benchmark_support)
    add_executable(vectorforge_topk_benchmark benchmarks/topk_main.cpp)
    target_link_libraries(vectorforge_topk_benchmark PRIVATE vectorforge_benchmark_support)
    add_executable(vectorforge_reuse_benchmark benchmarks/reuse_main.cpp)
    target_link_libraries(vectorforge_reuse_benchmark PRIVATE vectorforge_benchmark_support)
    add_executable(vectorforge_similarity_benchmark benchmarks/similarity_main.cpp)
    target_link_libraries(vectorforge_similarity_benchmark PRIVATE vectorforge_benchmark_support)
    set(vectorforge_benchmark_targets vectorforge_similarity_kernel vectorforge_benchmark_support vectorforge_benchmark vectorforge_topk_benchmark vectorforge_reuse_benchmark vectorforge_similarity_benchmark)
    if(BUILD_TESTING)
        add_executable(vectorforge_benchmark_tests tests/benchmark_test.cpp)
        target_link_libraries(vectorforge_benchmark_tests PRIVATE vectorforge_benchmark_support)
        list(APPEND vectorforge_benchmark_targets vectorforge_benchmark_tests)
        add_test(NAME benchmark_configuration_and_csv COMMAND vectorforge_benchmark_tests)
        add_test(NAME benchmark_smoke COMMAND vectorforge_benchmark --smoke)
        add_executable(vectorforge_topk_tests tests/topk_benchmark_test.cpp)
        target_link_libraries(vectorforge_topk_tests PRIVATE vectorforge_benchmark_support)
        list(APPEND vectorforge_benchmark_targets vectorforge_topk_tests)
        add_test(NAME topk_algorithms_and_csv COMMAND vectorforge_topk_tests)
        add_test(NAME topk_benchmark_smoke COMMAND vectorforge_topk_benchmark --smoke)
        add_executable(vectorforge_reuse_tests tests/reuse_benchmark_test.cpp)
        target_link_libraries(vectorforge_reuse_tests PRIVATE vectorforge_benchmark_support)
        list(APPEND vectorforge_benchmark_targets vectorforge_reuse_tests)
        add_test(NAME reuse_model_and_csv COMMAND vectorforge_reuse_tests)
        add_test(NAME reuse_benchmark_smoke COMMAND vectorforge_reuse_benchmark --smoke)
        add_executable(vectorforge_similarity_tests tests/similarity_benchmark_test.cpp)
        target_link_libraries(vectorforge_similarity_tests PRIVATE vectorforge_benchmark_support)
        list(APPEND vectorforge_benchmark_targets vectorforge_similarity_tests)
        add_test(NAME similarity_kernel_and_csv COMMAND vectorforge_similarity_tests)
        add_test(NAME similarity_benchmark_smoke COMMAND vectorforge_similarity_benchmark --smoke)
    endif()
    if(VECTORFORGE_BUILD_FMA_COMPARISON)
        if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$" OR
           NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$" OR MSVC)
            message(FATAL_ERROR "FMA comparison requires GCC/Clang targeting x86-64")
        endif()
        foreach(fma_variant generic hardware)
            set(fma_target vectorforge_fma_${fma_variant}_kernel)
            add_library(${fma_target} STATIC benchmarks/similarity_kernel.cpp)
            target_link_libraries(${fma_target} PUBLIC VectorForge::vectorforge)
            set_target_properties(${fma_target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
            target_compile_definitions(${fma_target} PRIVATE
                VECTORFORGE_SIMILARITY_NAMESPACE=vectorforge::benchmark::similarity::${fma_variant}_target)
            target_compile_options(${fma_target} PRIVATE -march=x86-64 -mtune=generic -fno-fast-math)
            if(fma_variant STREQUAL "generic")
                target_compile_options(${fma_target} PRIVATE -mno-avx -mno-avx2 -mno-fma)
            else()
                target_compile_options(${fma_target} PRIVATE -mavx -mno-avx2 -mfma)
            endif()
            if(VECTORFORGE_SIMILARITY_DIAGNOSTICS)
                target_compile_options(${fma_target} PRIVATE
                    "-fopt-info-vec-all=${CMAKE_CURRENT_BINARY_DIR}/fma-${fma_variant}-vectorization-$<CONFIG>.txt"
                    -save-temps=obj)
            endif()
            list(APPEND vectorforge_benchmark_targets ${fma_target})
        endforeach()
        add_library(vectorforge_fma_support STATIC benchmarks/fma_comparison.cpp)
        target_link_libraries(vectorforge_fma_support PUBLIC vectorforge_benchmark_support
            vectorforge_fma_generic_kernel vectorforge_fma_hardware_kernel)
        target_include_directories(vectorforge_fma_support PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/benchmark-info/$<CONFIG>")
        file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/benchmark-info/$<CONFIG>/fma_build_info.hpp" CONTENT
"#pragma once
namespace vectorforge::benchmark::similarity::fma::build_info {
inline constexpr char generic_options[] = R\"vf($<JOIN:$<TARGET_PROPERTY:vectorforge_fma_generic_kernel,COMPILE_OPTIONS>, >)vf\";
inline constexpr char hardware_options[] = R\"vf($<JOIN:$<TARGET_PROPERTY:vectorforge_fma_hardware_kernel,COMPILE_OPTIONS>, >)vf\";
}
")
        add_executable(vectorforge_fma_benchmark benchmarks/fma_main.cpp)
        target_link_libraries(vectorforge_fma_benchmark PRIVATE vectorforge_fma_support)
        list(APPEND vectorforge_benchmark_targets vectorforge_fma_support vectorforge_fma_benchmark)
        if(BUILD_TESTING)
            add_executable(vectorforge_fma_tests tests/fma_comparison_test.cpp)
            target_link_libraries(vectorforge_fma_tests PRIVATE vectorforge_fma_support)
            list(APPEND vectorforge_benchmark_targets vectorforge_fma_tests)
            add_test(NAME fma_numerical_equivalence_and_csv COMMAND vectorforge_fma_tests)
            add_test(NAME fma_benchmark_smoke COMMAND vectorforge_fma_benchmark --smoke)
            set_tests_properties(fma_numerical_equivalence_and_csv fma_benchmark_smoke PROPERTIES SKIP_RETURN_CODE 77)
        endif()
    endif()
    foreach(benchmark_target IN LISTS vectorforge_benchmark_targets)
        set_target_properties(${benchmark_target} PROPERTIES CXX_EXTENSIONS OFF)
        if(MSVC)
            target_compile_options(${benchmark_target} PRIVATE /W4 /permissive-)
        else()
            target_compile_options(${benchmark_target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
        endif()
    endforeach()
endif()
