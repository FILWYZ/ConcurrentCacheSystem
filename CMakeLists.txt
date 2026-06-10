cmake_minimum_required(VERSION 3.16)

project(CacheSystem LANGUAGES CXX)

add_library(cache_system INTERFACE)
target_include_directories(cache_system INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(cache_system INTERFACE cxx_std_17)

find_package(Threads REQUIRED)

add_executable(lfu_tests test_lfu.cpp)
target_link_libraries(lfu_tests PRIVATE cache_system Threads::Threads)

add_executable(arc_tests ARC/test_arc.cpp)
target_link_libraries(arc_tests PRIVATE cache_system Threads::Threads)

add_executable(cache_benchmark cache_benchmark.cpp)
target_link_libraries(cache_benchmark PRIVATE cache_system Threads::Threads)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(lfu_tests PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(arc_tests PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(cache_benchmark PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_test(NAME lfu_tests COMMAND lfu_tests)
add_test(NAME arc_tests COMMAND arc_tests)
