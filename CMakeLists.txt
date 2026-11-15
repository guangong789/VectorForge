cmake_minimum_required(VERSION 3.20)
project(VectorForge VERSION 0.1.0 LANGUAGES CXX)

option(VECTORFORGE_ENABLE_SANITIZERS "Enable AddressSanitizer and UBSan" OFF)

add_library(vectorforge
    src/column_batch.cpp
    src/embedding_buffer.cpp
    src/batch_view.cpp
    src/scalar_reference.cpp
    src/batched_executor.cpp
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
endif()
