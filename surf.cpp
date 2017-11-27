#include "algorithms.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

void readme();

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        readme();
        return -1;
    }

    std::vector<std::string> paths;
    paths.reserve(argc - 1);
    for (int i = 1; i < argc; i++)
    {
        paths.emplace_back(argv[i]);
    }

    std::vector<cv::Mat> images;
    readImages(paths, images);

    auto &img_1 = images[0];
    auto &img_2 = images[1];

    // Detecting the keypoints using SURF Detector
    double minHessian = 400.0;

    Ptr<SURF> detector = SURF::create(minHessian);
    std::vector<KeyPoint> keypoints_1, keypoints_2;

    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // Computing descriptors
    Mat descriptors1, descriptors2;
    detector->compute(img_1, keypoints_1, descriptors1);
    detector->compute(img_2, keypoints_2, descriptors2);

    // Matching descriptors
    Ptr<DescriptorMatcher> matcher = BFMatcher::create(cv::NORM_L2, true);
    std::vector<DMatch> matches;
    matcher->match(descriptors1, descriptors2, matches);

    // Removing unmatched keypoints
    removeUnmatched(keypoints_1, keypoints_2, matches);

    readExivMetadata(paths.front());

    // Drawing the results
    Mat img_keypoints;
    drawKeypoints(img_1, keypoints_1, img_keypoints);
    
    CallbackData data = {img_keypoints, keypoints_1, keypoints_2, matches, 5.0f, img_1.cols, 70.0f};
    namedWindow("matches", 1);
    setMouseCallback("matches", mouseCallback, &data);
    imshow("matches", img_keypoints);

    waitKey(0);

    return 0;
}

void readme() { std::cout << " Usage: ./surf <img1> <img2>" << std::endl; }