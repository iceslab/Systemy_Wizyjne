#ifndef _INCLUDE_ALGORITHMS_HPP_
#define _INCLUDE_ALGORITHMS_HPP_

#include <opencv2/core.hpp>
#include <cmath>
#include <tuple>
#include <algorithm>
#include <list>

typedef std::pair<cv::KeyPoint, cv::KeyPoint> keyPointPairT;

float euclideanDistance(float x1, float y1, float x2, float y2);
float euclideanDistance(const cv::KeyPoint & kp1, const cv::KeyPoint & kp2);

#endif // !_INCLUDE_ALGORITHMS_HPP_ 
