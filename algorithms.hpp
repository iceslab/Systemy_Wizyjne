#ifndef _INCLUDE_ALGORITHMS_HPP_
#define _INCLUDE_ALGORITHMS_HPP_

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>

#include <exiv2/exiv2.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <list>
#include <stdio.h>
#include <string>
#include <tuple>
#include <vector>

#define VERBOSITY_LEVEL 2
#include "Utilities/asserts.h"

typedef std::pair<cv::KeyPoint, cv::KeyPoint> keypointsPairT;

struct CallbackData
{
    cv::Mat &image;
    std::vector<cv::KeyPoint> &keypoints_1;
    std::vector<cv::KeyPoint> &keypoints_2;
    std::vector<cv::DMatch> &matches;
    float lensesDistance;
    int imageWidth;
    float cameraHorizontalAngle;
};

bool readImages(const std::vector<std::string> &paths, std::vector<cv::Mat> &images);

float euclideanDistance(float x1, float y1, float x2, float y2);
float euclideanDistance(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2);

void mouseCallback(int event, int x, int y, int flags, void *userdata);

float objectDistance(float lensesDistance, int imageWidth, float cameraHorizontalAngle,
                     const cv::KeyPoint &kp1, const cv::KeyPoint &kp2);

void readExivMetadata(std::string path);

std::vector<keypointsPairT> extractMatchedPairs(const std::vector<cv::KeyPoint> &keypoints_1,
                                                const std::vector<cv::KeyPoint> &keypoints_2,
                                                const std::vector<cv::DMatch> &matches);

void removeUnmatched(std::vector<cv::KeyPoint> &keypoints_1,
                     std::vector<cv::KeyPoint> &keypoints_2,
                     const std::vector<cv::DMatch> &matches);

#endif // !_INCLUDE_ALGORITHMS_HPP_
