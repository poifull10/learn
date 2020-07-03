find_package(OpenCV REQUIRED)

add_executable(ndt_match main.cpp)
target_link_libraries(ndt_match ${OpenCV_LIBS})
