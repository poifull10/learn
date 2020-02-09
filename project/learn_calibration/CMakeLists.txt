find_package(OpenCV REQUIRED)
project(calib)

find_package(OpenCV REQUIRED)
add_library(calib calibration/target_detector.cpp)
target_include_directories(calib PUBLIC include)
target_link_libraries(calib ${OpenCV_LIBS})

add_executable(calibtest tests/main.cpp)
target_link_libraries(calibtest calib gtest)

add_executable(run_calib main.cpp)
target_link_libraries(run_calib calib)