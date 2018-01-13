cmake_minimum_required (VERSION 2.8.11)
project (surf)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14")

add_executable(surf surf.cpp algorithms.cpp algorithms.hpp Utilities/asserts.h NormalDistribution.cpp NormalDistribution.h)
target_link_libraries(surf opencv_imgcodecs opencv_imgproc opencv_features2d opencv_xfeatures2d opencv_highgui opencv_core exiv2 opencv_calib3d)
