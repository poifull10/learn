add_subdirectory(eigen3)

add_library(Eigen3 INTERFACE)
target_include_directories(Eigen3 INTERFACE ${learn_SOURCE_DIR}/third_party/eigen3)

add_subdirectory(gtest)
