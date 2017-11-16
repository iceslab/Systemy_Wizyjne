#include "algorithms.hpp"

float euclideanDistance(float x1, float y1, float x2, float y2)
{
    const auto xDiff = fabs(x1 - x2);
    const auto yDiff = fabs(y1 - y2);
    return sqrt(xDiff * xDiff + yDiff * yDiff);
}

float euclideanDistance(const cv::KeyPoint & kp1, const cv::KeyPoint & kp2)
{
    return euclideanDistance(kp1.pt.x, kp1.pt.y, kp2.pt.x, kp2.pt.y);
}
