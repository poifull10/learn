set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5 COMPONENTS Widgets Qml Quick REQUIRED)

qt5_wrap_cpp(moc_sources data.h)

add_executable(qt_example
    main.cpp
    qml.qrc
    ${qml_QRC}
    ${moc_sources}
)


target_include_directories(qt_example SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/project/learn_qt )
target_link_libraries(qt_example Qt5::Widgets Qt5::Qml Qt5::Quick)