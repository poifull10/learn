add_library(kflib src/generateData.cpp src/kalmanFilter.cpp src/model.cpp)
target_include_directories(kflib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(kflib Eigen3)

add_executable(kfmain main.cpp)
target_link_libraries(kfmain kflib)

add_executable(kftest tests/main.cpp tests/testModel.cpp)
target_link_libraries(kftest kflib gtest)
