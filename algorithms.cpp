#include "algorithms.hpp"

bool readImages(const std::vector<std::string> &paths, std::vector<cv::Mat> &images)
{
    auto retVal = true;
    for (const auto &path : paths)
    {
        images.emplace_back(cv::imread(path.c_str(), cv::IMREAD_GRAYSCALE));
        if (!images.back().data)
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

float euclideanDistance(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2)
{
    return euclideanDistance(kp1.pt.x, kp1.pt.y, kp2.pt.x, kp2.pt.y);
}

void mouseClickCallback(int event, int x, int y, int flags, void *userdata)
{
    if (event != cv::EVENT_LBUTTONDOWN)
    {
        return;
    }

    auto data = reinterpret_cast<CallbackData *>(userdata);
    std::vector<float> distances;
    distances.reserve(data->keypoints_1.size());

    for (const auto &el : data->keypoints_1)
    {
        distances.emplace_back(euclideanDistance(x, y, el.pt.x, el.pt.y));
    }

    const auto it = std::min_element(distances.begin(), distances.end());
    size_t index = it - distances.begin();
    DEBUG_PRINTLN("index: %zu\n", index);

    std::cerr << "  LMB clicked - position (" << x << ", " << y << ")\n"
              << "Closest match - position (" << data->keypoints_1[index].pt.x << ", "
              << data->keypoints_1[index].pt.y << ")\n";

    const auto matchIt =
        std::find_if(data->matches.begin(), data->matches.end(),
                     [index](cv::DMatch el) -> bool { return el.queryIdx == index; });
    if (matchIt == data->matches.end())
    {
        std::cerr << "Match distance from camera: No match found for given point\n" << std::endl;
        return;
    }

    DEBUG_PRINTLN("matchIt->queryIdx: %d, matchIt->trainIdx: %d", matchIt->queryIdx,
                  matchIt->trainIdx);
    DEBUG_PRINTLN("keypoints_1.size(): %zu, keypoints_2.size(): %zu", data->keypoints_1.size(),
                  data->keypoints_2.size());
    const auto &kp1 = data->keypoints_1[matchIt->queryIdx];
    const auto &kp2 = data->keypoints_2[matchIt->trainIdx];

    const auto distanceFromCamera = objectDistance(data->lensesDistance, data->imageWidth,
                                                   data->cameraHorizontalAngle, kp1, kp2);
    std::cerr << "Match distance from camera: " << distanceFromCamera << "\n" << std::endl;
}

float objectDistance(float lensesDistance, int imageWidth, float cameraHorizontalAngle,
                     const cv::KeyPoint &kp1, const cv::KeyPoint &kp2)
{
    // D = (B * x_0) / (2 * tan(fi_0 / 2) * (x_L - x_R))
    //
    // D    - distance to object
    // B    - distance between lenses
    // x_0  - horizontal picture resolution (in pixels)
    // fi_0 - camera's horizontal angle
    // x_L  - left object position (in pixels)
    // x_R  - right object position (in pixels)

    DEBUG_PRINTLN("B: %6.3f x_0: %6.3f fi_0: %6.3f diff: %6.3f\n", lensesDistance,
                  static_cast<float>(imageWidth), cameraHorizontalAngle,
                  euclideanDistance(kp1, kp2));

    return (lensesDistance * static_cast<float>(imageWidth)) /
           (2.0f * tanf(cameraHorizontalAngle / 2.0f) * euclideanDistance(kp1, kp2));
}

void readExivMetadata(std::string path)
{
    Exiv2::Image::AutoPtr image = Exiv2::ImageFactory::open(path);
    image->readMetadata();
    Exiv2::ExifData &exifData = image->exifData();

    if (exifData.empty())
    {
        std::string error(path);
        error += ": No Exif data found in the file";
        std::cerr << error << std::endl;
        return;
    }

    Exiv2::ExifData::const_iterator end = exifData.end();

    for (Exiv2::ExifData::const_iterator i = exifData.begin(); i != end; ++i)
    {
        const char *tn = i->typeName();
        std::cout << std::setw(44) << std::setfill(' ') << std::left << i->key() << " "
                  << "0x" << std::setw(4) << std::setfill('0') << std::right << std::hex << i->tag()
                  << " " << std::setw(9) << std::setfill(' ') << std::left << (tn ? tn : "Unknown")
                  << " " << std::dec << std::setw(3) << std::setfill(' ') << std::right
                  << i->count() << "  " << std::dec << i->value() << "\n";
    }
}
