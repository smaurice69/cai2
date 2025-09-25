cmake_minimum_required(VERSION 3.20)
project(Chiron VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

enable_testing()

add_library(chiron_lib
    src/attacks.cpp
    src/bitboard.cpp
    src/board.cpp
    src/movegen.cpp
    src/perft.cpp
    src/search.cpp
    src/uci.cpp
    src/zobrist.cpp
    src/notation.cpp
    eval/evaluation.cpp
    nnue/network.cpp
    nnue/evaluator.cpp
    training/selfplay.cpp
    training/trainer.cpp
    training/pgn_importer.cpp
    tools/tuning.cpp
    tools/teacher.cpp
)

target_include_directories(chiron_lib
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/eval
        ${PROJECT_SOURCE_DIR}/nnue
        ${PROJECT_SOURCE_DIR}/training
        ${PROJECT_SOURCE_DIR}/tools
)

if (MSVC)
    target_compile_options(chiron_lib PRIVATE /W4 /permissive- /EHsc)
    target_compile_options(chiron_lib PRIVATE $<$<CONFIG:Release>:/O2 /Ot /Oi /GL /DNDEBUG>)
else()
    target_compile_options(chiron_lib PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(chiron_lib PRIVATE $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG -flto -fno-omit-frame-pointer>)
endif()

add_executable(chiron src/main.cpp)
target_link_libraries(chiron PRIVATE chiron_lib)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# Prevent overriding the parent project's compiler/linker settings on Windows.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(chiron_tests
    tests/test_perft.cpp
    tests/test_nnue.cpp
    tests/test_selfplay.cpp
    tests/test_training.cpp
)
target_link_libraries(chiron_tests PRIVATE chiron_lib GTest::gtest_main)

if (MSVC)
    target_link_options(chiron PRIVATE $<$<CONFIG:Release>:/LTCG>)
    target_link_options(chiron_tests PRIVATE $<$<CONFIG:Release>:/LTCG>)
else()
    target_link_options(chiron PRIVATE $<$<CONFIG:Release>:-flto>)
    target_link_options(chiron_tests PRIVATE $<$<CONFIG:Release>:-flto>)
endif()

include(GoogleTest)
gtest_discover_tests(chiron_tests)

