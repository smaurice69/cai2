cmake_minimum_required(VERSION 3.20)
project(Chiron VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
    eval/evaluation.cpp
)

target_include_directories(chiron_lib
    PUBLIC
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/eval
)

if (MSVC)
    target_compile_options(chiron_lib PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(chiron_lib PRIVATE -Wall -Wextra -Wpedantic)
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

add_executable(chiron_tests tests/test_perft.cpp)
target_link_libraries(chiron_tests PRIVATE chiron_lib GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(chiron_tests)

