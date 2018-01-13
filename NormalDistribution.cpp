#include "NormalDistribution.h"

namespace distribution
{

NormalDistribution::NormalDistribution(double mean, double stddev)
    : mean(mean), stddev(stddev), variance(stddev * stddev)
{
    // Nohing to do
}

double NormalDistribution::getProbabilityDenisty(double x) const
{
    const double exponent = (-((x - mean) * (x - mean)) / (2.0L * variance));
    const double retVal = (1.0 / (sqrt(2.0 * M_PI) * stddev)) * pow(M_E, exponent);
    return retVal;
}

double NormalDistribution::getMean() const { return mean; }

double NormalDistribution::getStddev() const { return stddev; }

double NormalDistribution::getVariance() const { return variance; }

double NormalDistribution::calculateMean(const std::vector<double> &data)
{
    const auto dataSize = data.size();
    double dataSum = 0.0;

    // Calculating mean
    for (const auto &currentValue : data)
    {
        dataSum += currentValue;
    }
    return dataSum / static_cast<double>(dataSize);
}

double NormalDistribution::calculateStddev(const std::vector<double> &data,
                                           const double calculatedMean)
{
    const auto dataSize = data.size();
    double dataSum = 0.0;

    // Calculating stddev
    for (const auto &currentValue : data)
    {
        dataSum += std::pow(std::abs(currentValue - calculatedMean), 2.0);
    }
    return sqrt(dataSum / static_cast<double>(dataSize));
}

double NormalDistribution::calculateStddev(const std::vector<double> &data)
{
    const double calculatedMean = calculateMean(data);
    return calculateStddev(data, calculatedMean);
}

double NormalDistribution::calculateVariance(const std::vector<double> &data)
{
    const double stddev = calculateStddev(data);
    return stddev * stddev;
}

double NormalDistribution::getDistributionInX(const std::vector<double> &data, double percent) const
{
    ASSERT(percent > 0.0 && percent <= 1.0);
    std::vector<double> valueProbability;
    valueProbability.reserve(data.size());
    double valueProbabilitySum = 0.0;
    for (const auto &el : data)
    {
        valueProbability.emplace_back(getProbabilityDenisty(el));
        valueProbabilitySum += valueProbability.back();
    }

    percent *= valueProbabilitySum;

    double probabilitySum = 0.0;
    size_t index = 0;
    std::sort(valueProbability.begin(), valueProbability.end());
    for (; index < data.size(); index++)
    {
        probabilitySum += valueProbability[index];
        if (probabilitySum >= percent)
        {
            break;
        }
    }
    return valueProbability[index];
}

NormalDistribution getNormalDistribution(const std::vector<double> &data)
{
    if (data.empty())
    {
        DEBUG_PRINTLN("%s", "Vector is empty");
        return NormalDistribution(0.0, 0.0);
    }

    const auto mean = NormalDistribution::calculateMean(data);
    const auto stddev = NormalDistribution::calculateStddev(data, mean);

    DEBUG_PRINTLN("data.size(): %zu mean: %f stddev: %f", data.size(), mean, stddev);
    return NormalDistribution(mean, stddev);
}
}
