#include "algorithms.hpp"

bool readImages(const std::vector<std::string> & paths, std::vector<cv::Mat> & images)
{
    auto retVal = true;
    for(const auto & path : paths)
    {
        images.emplace_back(cv::imread(path.c_str(), cv::IMREAD_GRAYSCALE));
        if(!images.back().data)
        {
            std::cerr << " --(!) Error reading images: " << path << std::endl; 
            retVal = false;
            images.clear();
            break;
        }
    }
    return retVal;    
}

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

void mouseClickCallback(int event, int x, int y, int flags, void* userdata)
{
    if(event != cv::EVENT_LBUTTONDOWN)
    {
        return;
    }
    
    auto data = reinterpret_cast<CallbackData*>(userdata);
    std::vector<float> distances;
    distances.reserve(data->keypoints_1.size());

    for(const auto& el : data->keypoints_1)
    {
        distances.emplace_back(euclideanDistance(x, y, el.pt.x, el.pt.y));
    }

    const auto it = std::min_element(distances.begin(), distances.end());
    size_t index = it - distances.begin();

    const auto matchIt = std::find_if(data->matches.begin(),
                                      data->matches.end(),
                                      [index](cv::DMatch el)->bool
        {
            return el.queryIdx == index;
        });
    const auto& kp1 = data->keypoints_1[matchIt->queryIdx];
    const auto& kp2 = data->keypoints_2[matchIt->trainIdx];

    const auto distanceFromCamera = objectDistance(5.0f, 500, 90.0f, kp1, kp2);
    std::cerr << "  LMB clicked - position (" << x << ", " << y << ")\n" 
              << "Closest match - position (" << kp1.pt.x << ", " << kp1.pt.y << ")\n" 
              << "Match distance from camera: " << distanceFromCamera << "\n" 
              << std::endl;
}

float objectDistance(float lensesDistance,
                     int imageWidth,
                     float cameraHorizontalAngle,
                     const cv::KeyPoint & kp1,
                     const cv::KeyPoint & kp2)
{
    // D = (B * x_0) / (2 * tan(fi_0 / 2) * (x_L - x_R))
    //
    // D    - distance to object
    // B    - distance between lenses
    // x_0  - horizontal picture resolution (in pixels)
    // fi_0 - camera's horizontal angle
    // x_L  - left object position (in pixels)
    // x_R  - right object position (in pixels)

    return (lensesDistance * static_cast<float>(imageWidth)) / 
           (2.0f * tanf(cameraHorizontalAngle / 2.0f) * 
           euclideanDistance(kp1, kp2));
}