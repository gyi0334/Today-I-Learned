cmake_minimum_required(VERSION 3.15)

project(OpenCVStudy)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(OpenCVStudy test.cpp)

target_link_libraries(OpenCVStudy PRIVATE ${OpenCV_LIBS})