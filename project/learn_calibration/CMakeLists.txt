find_package(OpenCV REQUIRED)

add_library(calib calibration/find_target.cpp)
target_include_directories(calib PUBLIC calibration)
target_link_libraries(calib ${OpenCV_LIBS})

add_executable(calibtest tests/main.cpp)
target_link_libraries(calibtest calib gtest)
