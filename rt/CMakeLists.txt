cmake_minimum_required(VERSION 3.20)
project(roboemot_rt CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
add_compile_options(-Wall -Wextra -Wpedantic -Werror)

add_library(roboemot_rt src/motion.cpp)
target_include_directories(roboemot_rt PUBLIC include)

enable_testing()
add_executable(test_golden tests/test_golden.cpp)
target_link_libraries(test_golden PRIVATE roboemot_rt)
add_test(NAME golden_motion
         COMMAND test_golden ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden_motion.txt)
add_executable(test_limits tests/test_limits.cpp)
target_link_libraries(test_limits PRIVATE roboemot_rt)
add_test(NAME limits COMMAND test_limits)
