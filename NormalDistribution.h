#ifndef _INCLUDE_NORMAL_DISTRIBUTION_H_
#define _INCLUDE_NORMAL_DISTRIBUTION_H_

#include <vector>
#include <algorithm>
#include <tuple>
#define USE_MATH_DEFINES
#include <cmath>
#define VERBOSITY_LEVEL 2
#include "Utilities/asserts.h"

namespace distribution
{
    class NormalDistribution
    {
    public:
        NormalDistribution(double mean, double stddev);
        ~NormalDistribution() = default;

        double getProbabilityDenisty(double x) const;
        double getMean() const;
        double getStddev() const;
        double getVariance() const;

        static double calculateMean(const std::vector<double> & data);
        static double calculateStddev(const std::vector<double> & data, const double calculatedMean);
        static double calculateStddev(const std::vector<double> & data);
        static double calculateVariance(const std::vector<double> & data);

        double getDistributionInX(const std::vector<double> & data, double percent) const;

    private:
        const double mean;
        const double stddev;
        const double variance;
    };

    NormalDistribution getNormalDistribution(const std::vector<double> & data);
}
#endif // !_INCLUDE_NORMAL_DISTRIBUTION_H_
