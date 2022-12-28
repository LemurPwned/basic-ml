/*
This is based on the code from
ByteTrack repository:
    https://github.com/ifzhang/ByteTrack/blob/main/yolox/tracker/kalman_filter.py
*/
#ifndef KALMAN_HPP
#define KALMAN_HPP
#include <iostream>
#include <vector>
#include <tuple>

#if defined __GNUC__ || defined __APPLE__
#include <Eigen/Core>
#include <Eigen/Cholesky>
#else
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Cholesky>
#endif

typedef Eigen::Matrix<double, 1, 4> ParamVector4d; // col vector
typedef Eigen::Matrix<double, 1, 8> ParamVector8d; // col vector
typedef Eigen::Matrix<double, 4, 4> CovMatrix4d;
typedef Eigen::Matrix<double, 8, 8> CovMatrix8d;
// typedef std::tuple<ParamVector8d, CovMatrix8d> GaussParams8d;
// typedef std::tuple<ParamVector4d, CovMatrix4d> GaussParams4d;
typedef std::pair<ParamVector8d, CovMatrix8d> GaussParams8d;
typedef std::pair<ParamVector4d, CovMatrix4d> GaussParams4d;

std::vector<double> detection2KalmanParams(const std::vector<double> &detection)
{
    // compute centroid
    double centroid_x = (detection[0] + detection[2]) / 2;
    double centroid_y = (detection[1] + detection[3]) / 2;
    // compute width and height
    double width = detection[2] - detection[0];
    double height = detection[3] - detection[1];
    double aspectRatio = width / height;
    std::vector<double> centroidBox = {centroid_x, centroid_y, aspectRatio, height};
    return centroidBox;
}

class KalmanFilterTracker
{
private:
    // unsigned int dynamicParams = 4; // x, y
    // unsigned int timSteps = 1;
    double dt = 1.;
    CovMatrix8d motionMatrix = CovMatrix8d::Identity();
    Eigen::Matrix<double, 4, 8> updateMatrix = Eigen::Matrix<double, 4, 8>::Identity();
    double stdDeviationPos = 1. / 20;
    double stdDeviationVel = 1. / 160;

public:
    KalmanFilterTracker()
    {
        for (int i = 0; i < 4; i++)
        {
            this->motionMatrix(i, i + 4) = dt;
        }
    }

    GaussParams8d init(const std::vector<double> &detection)
    {
        const auto measurement = detection2KalmanParams(detection);
        ParamVector8d mean{
            measurement[0],
            measurement[1],
            measurement[2],
            measurement[3],
            0, 0, 0, 0};

        ParamVector8d vec{
            pow(2 * this->stdDeviationPos * measurement[3], 2),
            pow(2 * this->stdDeviationPos * measurement[3], 2),
            pow(1e-2, 2),
            pow(2 * this->stdDeviationPos * measurement[3], 2),
            pow(10 * this->stdDeviationVel * measurement[3], 2),
            pow(10 * this->stdDeviationVel * measurement[3], 2),
            pow(1e-5, 2),
            pow(10 * this->stdDeviationVel * measurement[3], 2)};

        CovMatrix8d covarianceMatrix = vec.asDiagonal();
        return GaussParams8d(mean, covarianceMatrix);
    }

    GaussParams8d predict(const ParamVector8d &mean, const CovMatrix8d &covarianceMatrix)
    {
        ParamVector8d vec = {
            pow(this->stdDeviationPos * mean(3), 2),
            pow(this->stdDeviationPos * mean(3), 2),
            pow(1e-2, 2),
            pow(this->stdDeviationPos * mean(3), 2),
            pow(this->stdDeviationVel * mean(3), 2),
            pow(this->stdDeviationVel * mean(3), 2),
            pow(1e-5, 2),
            pow(this->stdDeviationVel * mean(3), 2)};
        CovMatrix8d motionCovariance = vec.asDiagonal();
        auto newMean = mean * this->motionMatrix.transpose();
        auto newCovarianceMatrix = this->motionMatrix * covarianceMatrix * this->motionMatrix.transpose() + motionCovariance;
        return GaussParams8d(newMean, newCovarianceMatrix);
    }

    GaussParams4d project(const ParamVector8d &mean, const CovMatrix8d &covarianceMatrix)
    {
        ParamVector4d std = {
            pow(this->stdDeviationPos * mean(3), 2),
            pow(this->stdDeviationPos * mean(3), 2),
            pow(1e-1, 2),
            pow(this->stdDeviationPos * mean(3), 2),
        };
        CovMatrix4d innovationCovariance = std.asDiagonal();
        ParamVector4d newMean = mean * this->updateMatrix.transpose();
        CovMatrix4d newCovarianceMatrix = (this->updateMatrix * covarianceMatrix) * this->updateMatrix.transpose();
        return GaussParams4d(newMean, newCovarianceMatrix + innovationCovariance);
    }

    GaussParams8d update(const ParamVector8d &mean, const CovMatrix8d &covarianceMatrix,
                         const std::vector<double> &detection)
    {
        const auto measurement = detection2KalmanParams(detection);
        const ParamVector4d measurementVec = {
            measurement[0],
            measurement[1],
            measurement[2],
            measurement[3]};
        GaussParams4d params = this->project(mean, covarianceMatrix);
        Eigen::LLT<CovMatrix4d> luSolver;
        const auto b = (covarianceMatrix * this->updateMatrix.transpose()).transpose();
        // luSolver.compute(params.second);
        const auto kalmanGain = luSolver.compute(params.second).solve(b);
        const ParamVector4d innovation = measurementVec - params.first;
        const ParamVector8d newMean = mean + (innovation * kalmanGain);
        const auto newCovarianceMatrix = covarianceMatrix - (kalmanGain.transpose() * params.second) * kalmanGain;
        return GaussParams8d(newMean, newCovarianceMatrix);
    }
};

#endif // KALMAN_HPP
