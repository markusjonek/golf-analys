cmake_minimum_required(VERSION 3.19)

project(golf_analyzer)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV)

include_directories(include ${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} golf_analyzer.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})