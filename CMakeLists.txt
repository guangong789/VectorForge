cmake_minimum_required(VERSION 3.20)
project(VectorForge VERSION 0.1.0 LANGUAGES CXX)

option(VECTORFORGE_ENABLE_SANITIZERS "Enable AddressSanitizer and UBSan" OFF)
option(VECTORFORGE_BUILD_BENCHMARKS "Build the standalone benchmark and its smoke tests" OFF)
option(VECTORFORGE_SIMILARITY_DIAGNOSTICS "Save GCC reports and assembly for the benchmark-only similarity kernel" OFF)
option(VECTORFORGE_BUILD_FMA_COMPARISON "Build isolated GCC/Clang x86-64 scalar FMA target comparisons" OFF)
option(VECTORFORGE_BUILD_ACCUMULATOR_STUDY "Build the isolated four-accumulator numerical/performance study" OFF)
option(VECTORFORGE_BUILD_CROSS_ROW_STUDY "Build the isolated AVX2/FMA gather feasibility prototype" OFF)
option(VECTORFORGE_BUILD_AVX2_SCORING "Compile the private guarded AVX2 scoring candidate when supported" ON)
option(VECTORFORGE_TEST_SIMD_DISPATCH "Legacy test marker only; cannot enable SIMD in production scoring" OFF)
mark_as_advanced(VECTORFORGE_TEST_SIMD_DISPATCH)
if(VECTORFORGE_BUILD_CROSS_ROW_STUDY AND NOT VECTORFORGE_BUILD_BENCHMARKS)
    message(FATAL_ERROR "Cross-row study requires VECTORFORGE_BUILD_BENCHMARKS")
endif()
if(VECTORFORGE_BUILD_ACCUMULATOR_STUDY AND
   (NOT VECTORFORGE_BUILD_BENCHMARKS OR NOT VECTORFORGE_BUILD_FMA_COMPARISON))
    message(FATAL_ERROR "Accumulator study requires VECTORFORGE_BUILD_BENCHMARKS and VECTORFORGE_BUILD_FMA_COMPARISON")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$" AND NOT MSVC)
    # Inspect effective compiler macros, not flag substrings (later flags may
    # override earlier ones). Actual core TUs also include this guard to catch
    # target/transitive options added after configure-time probing.
    function(vectorforge_check_numeric_contract)
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/src")
        set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
        if(CMAKE_CONFIGURATION_TYPES)
            set(vf_contract_configs ${CMAKE_CONFIGURATION_TYPES})
        elseif(CMAKE_BUILD_TYPE)
            set(vf_contract_configs "${CMAKE_BUILD_TYPE}")
        else()
            # An unconfigured single-config build has no configuration flags.
            set(vf_contract_configs "VectorForgeDefault")
        endif()
        foreach(vf_contract_config IN LISTS vf_contract_configs)
            set(CMAKE_TRY_COMPILE_CONFIGURATION "${vf_contract_config}")
            unset(VECTORFORGE_NUMERIC_CONTRACT_OK CACHE)
            unset(VECTORFORGE_NUMERIC_CONTRACT_OK)
            check_cxx_source_compiles("#include \"numeric_contract.hpp\"\nint main() { return 0; }"
                VECTORFORGE_NUMERIC_CONTRACT_OK)
            if(NOT VECTORFORGE_NUMERIC_CONTRACT_OK)
                message(FATAL_ERROR "VectorForge numerical contract rejected compiler options for ${vf_contract_config}; disable fast/finite/unsafe math. See src/numeric_contract.hpp for coverage.")
            endif()
        endforeach()
    endfunction()
    vectorforge_check_numeric_contract()
endif()

add_library(vectorforge
    src/data/column_batch.cpp
    src/data/embedding_buffer.cpp
    src/data/batch_view.cpp
    src/execution/scalar_reference.cpp
    src/execution/batched_executor.cpp
    src/scoring/candidate_scoring.cpp
    src/scoring/candidate_dispatch.cpp
    src/data/validated_dataset.cpp
    src/execution/filtered_count.cpp
)
add_library(VectorForge::vectorforge ALIAS vectorforge)
target_include_directories(vectorforge
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
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

set(vectorforge_has_avx2_scoring OFF)
if(VECTORFORGE_BUILD_AVX2_SCORING AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND
   CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$" AND
   CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$" AND NOT MSVC)
    set(vectorforge_has_avx2_scoring ON)
    add_library(vectorforge_avx2_scoring OBJECT src/scoring/candidate_scoring_avx2.cpp)
    target_include_directories(vectorforge_avx2_scoring PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_features(vectorforge_avx2_scoring PRIVATE cxx_std_20)
    target_compile_options(vectorforge_avx2_scoring PRIVATE
        -march=x86-64 -mtune=generic -mavx2 -mfma -fno-lto
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
    set_target_properties(vectorforge_avx2_scoring PROPERTIES
        CXX_EXTENSIONS OFF INTERPROCEDURAL_OPTIMIZATION FALSE POSITION_INDEPENDENT_CODE ON)
    if(VECTORFORGE_ENABLE_SANITIZERS)
        target_compile_options(vectorforge_avx2_scoring PRIVATE
            -fsanitize=address,undefined -fno-omit-frame-pointer)
    endif()
    target_sources(vectorforge PRIVATE $<TARGET_OBJECTS:vectorforge_avx2_scoring>)
endif()
target_compile_definitions(vectorforge PRIVATE
    VECTORFORGE_HAS_AVX2_SCORING=$<BOOL:${vectorforge_has_avx2_scoring}>)

if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$" AND NOT MSVC)
    set_source_files_properties(src/scoring/candidate_dispatch.cpp PROPERTIES COMPILE_OPTIONS "-frounding-math;-fno-lto")
endif()

include(CTest)
if(VECTORFORGE_TEST_SIMD_DISPATCH AND NOT BUILD_TESTING)
    message(FATAL_ERROR "Legacy SIMD trial marker is restricted to BUILD_TESTING=ON validation builds")
endif()
if(BUILD_TESTING)
    add_executable(vectorforge_tests tests/data/data_foundation_test.cpp)
    target_link_libraries(vectorforge_tests PRIVATE VectorForge::vectorforge)
    set_target_properties(vectorforge_tests PROPERTIES CXX_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(vectorforge_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(vectorforge_tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
    endif()
    add_test(NAME data_foundation COMMAND vectorforge_tests)

    add_executable(vectorforge_scalar_tests tests/execution/scalar_reference_test.cpp)
    target_link_libraries(vectorforge_scalar_tests PRIVATE VectorForge::vectorforge)
    set_target_properties(vectorforge_scalar_tests PROPERTIES CXX_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(vectorforge_scalar_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(vectorforge_scalar_tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
    endif()
    add_test(NAME scalar_reference COMMAND vectorforge_scalar_tests)

    add_executable(vectorforge_batched_tests tests/execution/batched_executor_test.cpp)
    target_link_libraries(vectorforge_batched_tests PRIVATE VectorForge::vectorforge)
    set_target_properties(vectorforge_batched_tests PROPERTIES CXX_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(vectorforge_batched_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(vectorforge_batched_tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
    endif()
    add_test(NAME batched_executor COMMAND vectorforge_batched_tests)

    add_executable(vectorforge_dataset_tests tests/data/validated_dataset_test.cpp)
    target_link_libraries(vectorforge_dataset_tests PRIVATE VectorForge::vectorforge)
    set_target_properties(vectorforge_dataset_tests PROPERTIES CXX_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(vectorforge_dataset_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(vectorforge_dataset_tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
    endif()
    add_test(NAME validated_dataset COMMAND vectorforge_dataset_tests)

    add_executable(vectorforge_count_tests tests/execution/filtered_count_test.cpp)
    target_link_libraries(vectorforge_count_tests PRIVATE VectorForge::vectorforge)
    set_target_properties(vectorforge_count_tests PROPERTIES CXX_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(vectorforge_count_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(vectorforge_count_tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
    endif()
    add_test(NAME filtered_count COMMAND vectorforge_count_tests)

    add_executable(vectorforge_candidate_tests tests/execution/candidate_scoring_test.cpp)
    target_link_libraries(vectorforge_candidate_tests PRIVATE VectorForge::vectorforge)
    target_include_directories(vectorforge_candidate_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    set_target_properties(vectorforge_candidate_tests PROPERTIES CXX_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(vectorforge_candidate_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(vectorforge_candidate_tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
    endif()
    add_test(NAME bounded_candidate_scoring COMMAND vectorforge_candidate_tests)

    add_executable(vectorforge_simd_dispatch_tests tests/execution/simd_dispatch_test.cpp)
    target_link_libraries(vectorforge_simd_dispatch_tests PRIVATE VectorForge::vectorforge)
    target_include_directories(vectorforge_simd_dispatch_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_definitions(vectorforge_simd_dispatch_tests PRIVATE
        VECTORFORGE_TEST_SIMD_DISPATCH=$<BOOL:${VECTORFORGE_TEST_SIMD_DISPATCH}>)
    set_target_properties(vectorforge_simd_dispatch_tests PROPERTIES CXX_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(vectorforge_simd_dispatch_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(vectorforge_simd_dispatch_tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
    endif()
    add_test(NAME simd_dispatch_and_fp_gate COMMAND vectorforge_simd_dispatch_tests)

    # Narrow Linux/ELF diagnostic only: no production dependency on libdl or
    # provider inspection. Whole-static, cross and sanitizer builds do not claim
    # LD_PRELOAD coverage; ordinary generic/static library builds remain valid.
    set(vectorforge_provider_link_flags "${CMAKE_CXX_FLAGS} ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")
    foreach(provider_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        string(APPEND vectorforge_provider_link_flags
            " ${CMAKE_CXX_FLAGS_${provider_config}} ${CMAKE_EXE_LINKER_FLAGS_${provider_config}} ${CMAKE_SHARED_LINKER_FLAGS_${provider_config}}")
    endforeach()
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_EXECUTABLE_FORMAT STREQUAL "ELF" AND
       CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$" AND NOT CMAKE_CROSSCOMPILING AND
       NOT VECTORFORGE_ENABLE_SANITIZERS AND NOT vectorforge_provider_link_flags MATCHES "(^|[ ,])-static(-pie)?($|[ ,])")
        add_library(vectorforge_fmaf_errno_provider SHARED tests/execution/fmaf_errno_provider.cpp)
        target_compile_features(vectorforge_fmaf_errno_provider PRIVATE cxx_std_20)
        target_link_libraries(vectorforge_fmaf_errno_provider PRIVATE ${CMAKE_DL_LIBS})
        set_target_properties(vectorforge_fmaf_errno_provider PROPERTIES
            CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF)
        add_library(vectorforge_provider_scalar_control OBJECT src/execution/batched_executor.cpp)
        target_link_libraries(vectorforge_provider_scalar_control PRIVATE VectorForge::vectorforge)
        target_include_directories(vectorforge_provider_scalar_control PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_compile_definitions(vectorforge_provider_scalar_control PRIVATE
            execute_batched=execute_batched_scalar_control score_candidates=score_candidates_scalar)
        add_executable(vectorforge_math_provider_tests tests/execution/math_provider_test.cpp
            $<TARGET_OBJECTS:vectorforge_provider_scalar_control>)
        target_link_libraries(vectorforge_math_provider_tests PRIVATE VectorForge::vectorforge ${CMAKE_DL_LIBS})
        target_include_directories(vectorforge_math_provider_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
        add_dependencies(vectorforge_math_provider_tests vectorforge_fmaf_errno_provider)
        foreach(provider_target vectorforge_fmaf_errno_provider vectorforge_provider_scalar_control vectorforge_math_provider_tests)
            target_compile_options(${provider_target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
            set_target_properties(${provider_target} PROPERTIES CXX_EXTENSIONS OFF INTERPROCEDURAL_OPTIMIZATION FALSE)
        endforeach()
        add_test(NAME scalar_provider_normal COMMAND ${CMAKE_COMMAND} -E env --unset=LD_PRELOAD
            $<TARGET_FILE:vectorforge_math_provider_tests> --normal)
        add_test(NAME scalar_provider_substitution COMMAND ${CMAKE_COMMAND} -E env
            "LD_PRELOAD=$<TARGET_FILE:vectorforge_fmaf_errno_provider>" $<TARGET_FILE:vectorforge_math_provider_tests> --replacement)
        set_tests_properties(scalar_provider_normal scalar_provider_substitution PROPERTIES SKIP_RETURN_CODE 77)
    else()
        message(STATUS "Provider-substitution regression omitted: requires native, unsanitized, dynamically linked Linux/ELF tests")
    endif()
endif()

if(VECTORFORGE_BUILD_BENCHMARKS)
    add_library(vectorforge_similarity_kernel STATIC benchmarks/similarity/similarity_kernel.cpp)
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
    add_library(vectorforge_benchmark_support STATIC benchmarks/benchmark.cpp benchmarks/topk/topk.cpp benchmarks/reuse/reuse.cpp benchmarks/similarity/similarity.cpp)
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
    add_executable(vectorforge_topk_benchmark benchmarks/topk/topk_main.cpp)
    target_link_libraries(vectorforge_topk_benchmark PRIVATE vectorforge_benchmark_support)
    add_executable(vectorforge_reuse_benchmark benchmarks/reuse/reuse_main.cpp)
    target_link_libraries(vectorforge_reuse_benchmark PRIVATE vectorforge_benchmark_support)
    add_executable(vectorforge_similarity_benchmark benchmarks/similarity/similarity_main.cpp)
    target_link_libraries(vectorforge_similarity_benchmark PRIVATE vectorforge_benchmark_support)
    set(vectorforge_benchmark_targets vectorforge_similarity_kernel vectorforge_benchmark_support vectorforge_benchmark vectorforge_topk_benchmark vectorforge_reuse_benchmark vectorforge_similarity_benchmark)
    if(BUILD_TESTING)
        add_executable(vectorforge_benchmark_tests tests/benchmarks/benchmark_test.cpp)
        target_link_libraries(vectorforge_benchmark_tests PRIVATE vectorforge_benchmark_support)
        list(APPEND vectorforge_benchmark_targets vectorforge_benchmark_tests)
        add_test(NAME benchmark_configuration_and_csv COMMAND vectorforge_benchmark_tests)
        add_test(NAME benchmark_smoke COMMAND vectorforge_benchmark --smoke)
        add_executable(vectorforge_topk_tests tests/benchmarks/topk_benchmark_test.cpp)
        target_link_libraries(vectorforge_topk_tests PRIVATE vectorforge_benchmark_support)
        list(APPEND vectorforge_benchmark_targets vectorforge_topk_tests)
        add_test(NAME topk_algorithms_and_csv COMMAND vectorforge_topk_tests)
        add_test(NAME topk_benchmark_smoke COMMAND vectorforge_topk_benchmark --smoke)
        add_executable(vectorforge_reuse_tests tests/benchmarks/reuse_benchmark_test.cpp)
        target_link_libraries(vectorforge_reuse_tests PRIVATE vectorforge_benchmark_support)
        list(APPEND vectorforge_benchmark_targets vectorforge_reuse_tests)
        add_test(NAME reuse_model_and_csv COMMAND vectorforge_reuse_tests)
        add_test(NAME reuse_benchmark_smoke COMMAND vectorforge_reuse_benchmark --smoke)
        add_executable(vectorforge_similarity_tests tests/benchmarks/similarity_benchmark_test.cpp)
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
            add_library(${fma_target} STATIC benchmarks/similarity/similarity_kernel.cpp)
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
        add_library(vectorforge_fma_support STATIC benchmarks/similarity/fma_comparison.cpp)
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
        add_executable(vectorforge_fma_benchmark benchmarks/similarity/fma_main.cpp)
        target_link_libraries(vectorforge_fma_benchmark PRIVATE vectorforge_fma_support)
        list(APPEND vectorforge_benchmark_targets vectorforge_fma_support vectorforge_fma_benchmark)
        if(BUILD_TESTING)
            add_executable(vectorforge_fma_tests tests/benchmarks/fma_comparison_test.cpp)
            target_link_libraries(vectorforge_fma_tests PRIVATE vectorforge_fma_support)
            list(APPEND vectorforge_benchmark_targets vectorforge_fma_tests)
            add_test(NAME fma_numerical_equivalence_and_csv COMMAND vectorforge_fma_tests)
            add_test(NAME fma_benchmark_smoke COMMAND vectorforge_fma_benchmark --smoke)
            set_tests_properties(fma_numerical_equivalence_and_csv fma_benchmark_smoke PROPERTIES SKIP_RETURN_CODE 77)
        endif()
        if(VECTORFORGE_BUILD_ACCUMULATOR_STUDY)
            add_library(vectorforge_fourway_kernel STATIC benchmarks/similarity/fourway_kernel.cpp)
            target_link_libraries(vectorforge_fourway_kernel PUBLIC VectorForge::vectorforge)
            set_target_properties(vectorforge_fourway_kernel PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
            # Match the existing sequential hardware kernel's code-generation
            # settings; keep target flags private to this experiment.
            target_compile_options(vectorforge_fourway_kernel PRIVATE
                -march=x86-64 -mtune=generic -fno-fast-math -mavx -mno-avx2 -mfma)
            if(VECTORFORGE_SIMILARITY_DIAGNOSTICS)
                target_compile_options(vectorforge_fourway_kernel PRIVATE
                    "-fopt-info-vec-all=${CMAKE_CURRENT_BINARY_DIR}/fourway-vectorization-$<CONFIG>.txt"
                    -save-temps=obj)
            endif()
            add_library(vectorforge_accumulator_support STATIC benchmarks/similarity/accumulator.cpp)
            target_link_libraries(vectorforge_accumulator_support PUBLIC vectorforge_fma_support vectorforge_fourway_kernel)
            target_include_directories(vectorforge_accumulator_support PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/benchmark-info/$<CONFIG>")
            file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/benchmark-info/$<CONFIG>/accumulator_build_info.hpp" CONTENT
"#pragma once
namespace vectorforge::benchmark::accumulator::build_info {
inline constexpr char sequential_options[] = R\"vf($<JOIN:$<TARGET_PROPERTY:vectorforge_fma_hardware_kernel,COMPILE_OPTIONS>, >)vf\";
inline constexpr char fourway_options[] = R\"vf($<JOIN:$<TARGET_PROPERTY:vectorforge_fourway_kernel,COMPILE_OPTIONS>, >)vf\";
}
")
            add_executable(vectorforge_accumulator_benchmark benchmarks/similarity/accumulator_main.cpp)
            target_link_libraries(vectorforge_accumulator_benchmark PRIVATE vectorforge_accumulator_support)
            list(APPEND vectorforge_benchmark_targets vectorforge_fourway_kernel
                vectorforge_accumulator_support vectorforge_accumulator_benchmark)
            if(BUILD_TESTING)
                add_executable(vectorforge_accumulator_tests tests/benchmarks/accumulator_test.cpp)
                target_link_libraries(vectorforge_accumulator_tests PRIVATE vectorforge_accumulator_support)
                list(APPEND vectorforge_benchmark_targets vectorforge_accumulator_tests)
                add_test(NAME accumulator_numerics_tails_and_ranking COMMAND vectorforge_accumulator_tests)
                add_test(NAME accumulator_benchmark_smoke COMMAND vectorforge_accumulator_benchmark --smoke)
                add_test(NAME accumulator_ranking_analysis COMMAND vectorforge_accumulator_benchmark --ranking)
                set_tests_properties(accumulator_numerics_tails_and_ranking accumulator_benchmark_smoke
                    accumulator_ranking_analysis PROPERTIES SKIP_RETURN_CODE 77)
            endif()
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

    if(VECTORFORGE_BUILD_CROSS_ROW_STUDY)
        if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$" OR
           NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$" OR MSVC)
            message(FATAL_ERROR "Cross-row study requires GCC/Clang targeting x86-64")
        endif()
        foreach(variant scalar avx2)
            add_library(vectorforge_cross_row_${variant} STATIC
                benchmarks/similarity/cross_row_${variant}.cpp)
            target_link_libraries(vectorforge_cross_row_${variant} PUBLIC VectorForge::vectorforge)
            target_include_directories(vectorforge_cross_row_${variant} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
            target_compile_options(vectorforge_cross_row_${variant} PRIVATE
                -march=x86-64 -mtune=generic -mavx2 -mfma -fno-fast-math
                -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
            set_target_properties(vectorforge_cross_row_${variant} PROPERTIES
                INTERPROCEDURAL_OPTIMIZATION FALSE CXX_EXTENSIONS OFF)
        endforeach()
        target_link_libraries(vectorforge_cross_row_avx2 PRIVATE vectorforge_cross_row_scalar)
        # Driver/CPU guard and test oracle compile for the ordinary generic target.
        add_executable(vectorforge_cross_row_benchmark benchmarks/similarity/cross_row_main.cpp)
        target_link_libraries(vectorforge_cross_row_benchmark PRIVATE
            vectorforge_cross_row_scalar vectorforge_cross_row_avx2 vectorforge_benchmark_support)
        target_include_directories(vectorforge_cross_row_benchmark PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}/benchmark-info/$<CONFIG>")
        file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/benchmark-info/$<CONFIG>/cross_row_build_info.hpp" CONTENT
"#pragma once
namespace vectorforge::benchmark::cross_row {
inline constexpr char scalar_options[] = R\"vf($<JOIN:$<TARGET_PROPERTY:vectorforge_cross_row_scalar,COMPILE_OPTIONS>, >)vf\";
inline constexpr char avx2_options[] = R\"vf($<JOIN:$<TARGET_PROPERTY:vectorforge_cross_row_avx2,COMPILE_OPTIONS>, >)vf\";
}
")
        set(cross_row_drivers vectorforge_cross_row_benchmark)
        if(BUILD_TESTING)
            add_executable(vectorforge_cross_row_tests tests/benchmarks/cross_row_test.cpp)
            target_link_libraries(vectorforge_cross_row_tests PRIVATE
                vectorforge_cross_row_scalar vectorforge_cross_row_avx2)
            target_include_directories(vectorforge_cross_row_tests PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks" "${CMAKE_CURRENT_SOURCE_DIR}/src")
            list(APPEND cross_row_drivers vectorforge_cross_row_tests)
            add_test(NAME cross_row_numerics_mapping_and_bounds COMMAND vectorforge_cross_row_tests)
            add_test(NAME cross_row_smoke COMMAND vectorforge_cross_row_benchmark --smoke)
            set_tests_properties(cross_row_numerics_mapping_and_bounds cross_row_smoke
                PROPERTIES SKIP_RETURN_CODE 77)
        endif()
        foreach(driver IN LISTS cross_row_drivers)
            target_compile_options(${driver} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
            set_target_properties(${driver} PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE CXX_EXTENSIONS OFF)
        endforeach()

        # V1-07C is study-only and reuses the existing study opt-in. No production
        # dispatch/ISA configuration changes; no new production feature flag.
        if(BUILD_TESTING AND vectorforge_has_avx2_scoring AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
            add_library(vectorforge_fp_state_kernel STATIC benchmarks/similarity/fp_state_avx2.cpp)
            target_link_libraries(vectorforge_fp_state_kernel PUBLIC VectorForge::vectorforge)
            target_compile_options(vectorforge_fp_state_kernel PRIVATE -march=x86-64 -mtune=generic -mavx2 -mfma)
            add_library(vectorforge_fp_state_support STATIC benchmarks/similarity/fp_state.cpp)
            target_link_libraries(vectorforge_fp_state_support PUBLIC vectorforge_fp_state_kernel)
            add_executable(vectorforge_fp_state_study benchmarks/similarity/fp_state_main.cpp)
            target_link_libraries(vectorforge_fp_state_study PRIVATE vectorforge_fp_state_support ${CMAKE_DL_LIBS})
            target_compile_definitions(vectorforge_fp_state_study PRIVATE
                VECTORFORGE_FP_STUDY_CONFIG="$<CONFIG>" VECTORFORGE_FP_STUDY_SANITIZED=$<BOOL:${VECTORFORGE_ENABLE_SANITIZERS}>)
            foreach(study_target vectorforge_fp_state_kernel vectorforge_fp_state_support vectorforge_fp_state_study)
                target_include_directories(${study_target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
                target_compile_options(${study_target} PRIVATE -frounding-math -fno-lto
                    -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
                set_target_properties(${study_target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE CXX_EXTENSIONS OFF)
            endforeach()
            add_test(NAME fp_state_isolation_and_replay COMMAND vectorforge_fp_state_study)
            set_tests_properties(fp_state_isolation_and_replay PROPERTIES SKIP_RETURN_CODE 77)
        endif()
    endif()
endif()
