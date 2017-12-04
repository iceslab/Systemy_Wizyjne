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
    static auto targetKp1 = cv::KeyPoint(0, 0, 1);
    static auto targetKp2 = cv::KeyPoint(0, 0, 1);

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
    if (!mouseDown)
    {
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
    int labelX = ((x + size.width) > labeled_image.cols) ? labeled_image.cols - size.width : x;
    int labelY = y - size.height < 0 ? size.height : y;
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

Exiv2::ExifData readExivMetadata(const std::string &path)
{
    Exiv2::Image::AutoPtr image = Exiv2::ImageFactory::open(path);
    image->readMetadata();
    Exiv2::ExifData &exifData = image->exifData();

    if (exifData.empty())
    {
        std::string error(path);
        error += ": No Exif data found in the file";
        std::cerr << error << std::endl;
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

    return exifData;
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

float floatGetHfovFromFile(const std::string &path)
{
    DEBUG_PRINTLN("%s", "Getting metadata");
    auto retVal = std::numeric_limits<float>::max();

    const Exiv2::ExifKey imageWidthKey("Exif.Photo.PixelXDimension");
    const Exiv2::ExifKey imageLengthKey("Exif.Photo.PixelYDimension");

    const Exiv2::ExifKey focalLengthInMmKey("Exif.Photo.FocalLength");
    const Exiv2::ExifKey xResKey("Exif.Photo.FocalPlaneXResolution");
    const Exiv2::ExifKey yResKey("Exif.Photo.FocalPlaneYResolution");
    const Exiv2::ExifKey resUnitKey("Exif.Photo.FocalPlaneResolutionUnit");

    auto imageWidth = std::numeric_limits<long>::max();
    auto imageLength = std::numeric_limits<long>::max();

    auto focalLength = std::numeric_limits<double>::quiet_NaN();
    auto xRes = std::numeric_limits<double>::quiet_NaN();
    auto yRes = std::numeric_limits<double>::quiet_NaN();
    auto resUnit = std::numeric_limits<long>::max();

    const auto image = Exiv2::ImageFactory::open(path);
    image->readMetadata();
    const auto &exifData = image->exifData();

    if (!exifData.empty())
    {
        auto it = exifData.findKey(imageWidthKey);
        size_t keysFound = 0;
        size_t desiredKeys = 6;
        if (it != exifData.end())
        {
            keysFound++;
            imageWidth = it->value().toLong();
            DEBUG_PRINTLN("Found: %s %ld px", it->key().c_str(), imageWidth);
        }

        it = exifData.findKey(imageLengthKey);
        if (it != exifData.end())
        {
            keysFound++;
            imageLength = it->value().toLong();
            DEBUG_PRINTLN("Found: %s %ld px", it->key().c_str(), imageLength);
        }

        it = exifData.findKey(focalLengthInMmKey);
        if (it != exifData.end())
        {
            keysFound++;
            const auto tmp = it->value().toRational();
            focalLength = static_cast<double>(tmp.first) / static_cast<double>(tmp.second);
            DEBUG_PRINTLN("Found: %s %f mm", it->key().c_str(), focalLength);
        }

        it = exifData.findKey(xResKey);
        if (it != exifData.end())
        {
            keysFound++;
            const auto tmp = it->value().toRational();
            xRes = static_cast<double>(tmp.first) / static_cast<double>(tmp.second);
            DEBUG_PRINTLN("Found: %s %f", it->key().c_str(), xRes);
        }

        it = exifData.findKey(yResKey);
        if (it != exifData.end())
        {
            keysFound++;
            const auto tmp = it->value().toRational();
            yRes = static_cast<double>(tmp.first) / static_cast<double>(tmp.second);
            DEBUG_PRINTLN("Found: %s %f", it->key().c_str(), yRes);
        }

        it = exifData.findKey(resUnitKey);
        if (it != exifData.end())
        {
            keysFound++;
            resUnit = it->value().toLong();
            std::string name = "unrecognized";
            switch (resUnit)
            {
            case 2:
                name = "cm";
                break;
            case 3:
                name = "inch";
                break;
            }

            DEBUG_PRINTLN("Found: %s %s", it->key().c_str(), name.c_str());
        }

        if (keysFound < desiredKeys)
        {
            DEBUG_PRINTLN("Not enough exif keys found. %zu/%zu", keysFound, desiredKeys);
        }
        else
        {
            DEBUG_PRINTLN("%s", "All exif keys found!");

            if (resUnit == 2 || resUnit == 3)
            {
                float ratio;
                switch (resUnit)
                {
                case 2:
                    ratio = 10.0f; // Convert from cm to mm
                    break;
                case 3:
                    ratio = 25.4f; // Convert from inch to mm
                    break;
                }

                DEBUG_PRINTLN("imageWidth: %ld xRes: %f, ratio: %f", imageWidth, xRes, ratio);
                const auto h = (imageWidth / xRes) * ratio;
                DEBUG_PRINTLN("h: %f mm 2f: %f mm", h, 2.0f * focalLength);
                retVal = 2.0f * atan(h / (2.0f * focalLength)) * (180.0f / M_PI);
                DEBUG_PRINTLN("Calculated angle: %f deg", retVal);
            }
            else
            {
                DEBUG_PRINTLN("Unrecognized resolution unit (%ld)", resUnit);
            }
        }
    }
    else
    {
        DEBUG_PRINTLN("%s", "No metadata found");
    }

    return retVal;
}