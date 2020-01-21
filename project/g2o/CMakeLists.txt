add_executable(lrnG2o main.cpp)
target_link_libraries(lrnG2o core solver_cholmod types_sba)
target_include_directories(lrnG2o PUBLIC ${learn_SOURCE_DIR}/third_party/g2o)
