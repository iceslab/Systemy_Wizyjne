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

void mouseCallback(int event, int x, int y, int flags, void *userdata)
{
    static auto mouseDown = false;
    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN:
        mouseDown = true;
        break;
    case cv::EVENT_LBUTTONUP:
        mouseDown = false;
        break;
    case cv::EVENT_MOUSEMOVE:
        // Continue execution
        break;
    default:
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
    DEBUG_PRINTLN("index: %zu", index);
    DEBUG_PRINTLN("keypoints_1.size(): %zu, keypoints_2.size(): %zu", data->keypoints_1.size(),
                  data->keypoints_2.size());

    const auto &kp1 = data->keypoints_1[index];
    const auto &kp2 = data->keypoints_2[index];
    static auto targetKp1 = cv::KeyPoint(0, 0, 1);
    static auto targetKp2 = cv::KeyPoint(0, 0, 1);

    if (!mouseDown)
    {
        targetKp1 = kp1;
        targetKp2 = kp2;
    }

    const auto distanceFromCamera = objectDistance(
        data->lensesDistance, data->imageWidth, data->cameraHorizontalAngle, targetKp1, targetKp2);
    DEBUG_PRINTLN("Match distance from camera: %f\n", distanceFromCamera);
    std::stringstream ss;
    ss << distanceFromCamera;

    cv::Mat labeled_image;
    data->image.copyTo(labeled_image);
    cv::arrowedLine(labeled_image, cv::Point(x, y), targetKp1.pt, CV_RGB(255, 0, 0), 2, CV_AA, 0);
    int baseline = 0;
    const auto size =
        cv::getTextSize(ss.str(), LABEL_FONT_FACE, LABEL_FONT_SCALE, LABEL_THICKNESS, &baseline);
    int labelX = x + size.width > labeled_image.cols ? x - size.width : x;
    int labelY = y - size.height < 0 ? y + size.height : y;
    cv::putText(labeled_image, ss.str(), cv::Point(labelX, labelY), LABEL_FONT_FACE,
                LABEL_FONT_SCALE, CV_RGB(0, 255, 0), LABEL_THICKNESS, CV_AA);
    imshow("matches", labeled_image);
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

    DEBUG_PRINTLN("B: %6.3f x_0: %6.3f fi_0: %6.3f diff: %6.3f", lensesDistance,
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

std::vector<keypointsPairT> extractMatchedPairs(const std::vector<cv::KeyPoint> &keypoints_1,
                                                const std::vector<cv::KeyPoint> &keypoints_2,
                                                const std::vector<cv::DMatch> &matches)
{
    auto retVal = std::vector<keypointsPairT>();
    retVal.reserve(matches.size());

    for (const auto match : matches)
    {
        retVal.emplace_back(keypoints_1[match.queryIdx], keypoints_2[match.trainIdx]);
    }

    return retVal;
}

void removeUnmatched(std::vector<cv::KeyPoint> &keypoints_1, std::vector<cv::KeyPoint> &keypoints_2,
                     const std::vector<cv::DMatch> &matches)
{
    auto keypoints_1_tmp = std::vector<cv::KeyPoint>();
    keypoints_1_tmp.reserve(matches.size());
    auto keypoints_2_tmp = std::vector<cv::KeyPoint>();
    keypoints_2_tmp.reserve(matches.size());

    for (const auto match : matches)
    {
        keypoints_1_tmp.emplace_back(keypoints_1[match.queryIdx]);
        keypoints_2_tmp.emplace_back(keypoints_2[match.trainIdx]);
    }

    keypoints_1 = std::move(keypoints_1_tmp);
    keypoints_2 = std::move(keypoints_2_tmp);
}
