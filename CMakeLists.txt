cmake_minimum_required(VERSION 3.19)

project(obj_det)

set(CMAKE_CXX_STANDARD 14)

set(OpenCV_DIR "/opt/homebrew/Cellar/opencv")

find_package(OpenCV)

include_directories(include ${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})