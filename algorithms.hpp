#ifndef _INCLUDE_ALGORITHMS_HPP_
#define _INCLUDE_ALGORITHMS_HPP_

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>

#include <cmath>
#include <tuple>
#include <algorithm>
#include <list>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>

struct CallbackData
{
    std::vector<cv::KeyPoint>& keypoints_1;
    std::vector<cv::KeyPoint>& keypoints_2;
    std::vector<cv::DMatch>& matches;
};

bool readImages(const std::vector<std::string> & paths, std::vector<cv::Mat> & images);

float euclideanDistance(float x1, float y1, float x2, float y2);
float euclideanDistance(const cv::KeyPoint & kp1, const cv::KeyPoint & kp2);

void mouseClickCallback(int event, int x, int y, int flags, void* userdata);

float objectDistance(float lensesDistance,
                     int imageWidth,
                     float cameraHorizontalAngle,
                     const cv::KeyPoint & kp1,
                     const cv::KeyPoint & kp2);

#endif // !_INCLUDE_ALGORITHMS_HPP_ 
