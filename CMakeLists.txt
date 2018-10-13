cmake_minimum_required(VERSION 3.6)
project(RaspCar)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}-std=c++11")

set(OpenCV_DIR "C:\\opencv-3.3.0\\mingw64-build\\install")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}"${CMAKE_SOURCE_DIR}/cmake/")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp GPIOlib.h libGPIO.cpp)
add_executable(RaspCar ${SOURCE_FILES})

target_link_libraries(RaspCar ${OpenCV_LIBS})