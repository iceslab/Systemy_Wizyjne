#include "algorithms.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

void readme();

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        readme();
        return -1;
    }

    float camerasDistance;
    sscanf(argv[1], "%f", &camerasDistance);

    std::vector<std::string> paths;
    paths.reserve(argc - 2);
    for (int i = 2; i < argc; i++)
    {
        paths.emplace_back(argv[i]);
    }

    std::vector<cv::Mat> images;
    readImages(paths, images);

    if(images.size() < 2){
        readme();
        return -2;
    }

    const auto hfov = floatGetHfovFromFile(paths.front());

    const float desiredWidth = 1024.0f;
    const float ratio = desiredWidth / static_cast<float>(images[0].size().width);

    cv::Mat img_1;
    cv::Mat img_2;
    cv::resize(images[0], img_1, Size(), ratio, ratio);
    cv::resize(images[1], img_2, Size(), ratio, ratio);
    

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

    // const auto exifData = readExivMetadata(paths.front());
    // Exiv2::ExifKey ek("Exif.Photo.FocalLength");
    // const auto it = exifData.findKey(ek);
    // if(it != exifData.end()){
    //     it->value().toFloat();
    // }

    // Drawing the results
    Mat img_keypoints;
    drawKeypoints(img_1, keypoints_1, img_keypoints);
    
    CallbackData data = {img_keypoints, keypoints_1, keypoints_2, matches, 5.0f, img_1.size().width, 70.0f};
    namedWindow("matches", cv::WINDOW_AUTOSIZE);
    setMouseCallback("matches", mouseCallback, &data);
    imshow("matches", img_keypoints);

    waitKey(0);

    return 0;
}

void readme() { std::cout << " Usage: ./surf <camerasDistance> <img1> <img2> ..." << std::endl; }