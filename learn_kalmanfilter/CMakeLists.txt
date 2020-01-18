add_executable(kfmain main.cpp src/generateData.cpp src/kalmanFilter.cpp)
target_include_directories(kfmain PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(kfmain Eigen3)
