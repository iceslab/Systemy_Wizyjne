#include <stdio.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include "algorithms.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

void readme();

int main( int argc, char** argv )
{
  if( argc != 3 )
  { 
    readme(); 
    return -1; 
  }

  Mat img_1 = imread( argv[1], IMREAD_GRAYSCALE );
  Mat img_2 = imread( argv[2], IMREAD_GRAYSCALE );

  if( !img_1.data || !img_2.data )
  { 
    std::cout<< " --(!) Error reading images " << std::endl; 
    return -1; 
  }

  // Detecting the keypoints using SURF Detector
  int minHessian = 400;

  Ptr<SURF> detector = SURF::create( minHessian );
  std::vector<KeyPoint> keypoints_1, keypoints_2;

  detector->detect( img_1, keypoints_1 );
  detector->detect( img_2, keypoints_2 );

  // Computing descriptors
  Mat descriptors1, descriptors2;
  detector->compute(img_1, keypoints_1, descriptors1);
  detector->compute(img_2, keypoints_2, descriptors2);

  // Matching descriptors
  Ptr<DescriptorMatcher> matcher = BFMatcher::create(cv::NORM_L2, true);
  std::vector<DMatch> matches;
  matcher->match(descriptors1, descriptors2, matches);

  // Drawing the results
  namedWindow("matches", 1);
  Mat img_matches;
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);
  imshow("matches", img_matches);

  fprintf(stderr, "\n");
  for(const auto& el : matches)
  {
    const auto& matched_1 = keypoints_1[el.queryIdx];
    const auto& matched_2 = keypoints_2[el.trainIdx];
    fprintf(stderr, "%7.2f ", euclideanDistance(matched_1, matched_2));
  }
  fprintf(stderr, "\n");
  waitKey(0);

  return 0;
}

  /** @function readme */
void readme()
{ 
    std::cout << " Usage: ./surf <img1> <img2>" << std::endl; 
}