cmake_minimum_required(VERSION 3.10)
project(learn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#  add_compile_options(-Wall -Wextra -Wpedantic)
#endif()

add_subdirectory(third_party)
add_subdirectory(learn_lm)
add_subdirectory(learn_kalmanfilter)
