CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FIND_PACKAGE(OpenCV REQUIRED)

PROJECT(TP5)
include_directories( /usr/local/Cellar/eigen/3.3.7/include/eigen3 )

ADD_EXECUTABLE(homographie homographie.cpp image.cpp image.h)
TARGET_LINK_LIBRARIES(homographie ${OpenCV_LIBS})
