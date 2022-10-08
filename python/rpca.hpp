#ifndef RPCA_HPP
#define RPCA_HPP

#include <stdio.h>

#if defined __GNUC__ || defined __APPLE__
#include <Eigen/Dense>
#include <Eigen/SVD>
#else
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#endif

// using namespace Eigen;
using MatrixXRef = Eigen::Ref<Eigen::MatrixXd>;
using MatriXdR = Eigen::MatrixXd;
class RPCA
{

private:
    unsigned int maxCount = 1000;
    MatriXdR L, S, Y;
    double thresholdScale;
    // this is unstable with the fastmath, so be careful
    // when compiling. Maybe worth switching to JacobiSVD
    // BDCSVD<MatriXdR, ComputeThinU | ComputeThinV> svd();
public:
    explicit RPCA(unsigned int maxCount, double thresholdScale = 1e-7) : maxCount(maxCount),
                                                                         thresholdScale(thresholdScale) {}

    MatriXdR getL() const
    {
        return L;
    }

    MatriXdR getS() const
    {
        return S;
    }

    double getFrobeniusNorm(const MatriXdR &X)
    {
        // by default for the matrices, Eigen
        // computes the Frobenius norm
        return X.norm();
    }
    MatriXdR shrinkMatrix(const MatriXdR &X, double tau)
    {
        const MatriXdR tauM = MatriXdR::Constant(X.rows(), X.cols(), tau);
        const MatriXdR thrsM = (X.cwiseAbs() - tauM);
        if (thrsM.maxCoeff() > 0)
        {
            return thrsM.cwiseProduct(X.cwiseSign());
        }
        else
        {
            return MatriXdR::Zero(X.rows(), X.cols());
        }
    }
    MatriXdR truncatedSVD(const MatriXdR &X, double tau)
    {
        Eigen::BDCSVD<MatriXdR> svd(
            X, Eigen::ComputeThinV | Eigen::ComputeThinU);
        const MatriXdR Stmp = svd.singularValues().asDiagonal();
        // MatriXdR Stmp(X.rows(), X.cols());
        // Stmp << (svd.singularValues())[0], 0.0, 0.0, (svd.singularValues())[1];
        const auto shrunkM = shrinkMatrix(Stmp, tau);
        return svd.matrixU() * shrunkM * svd.matrixV().transpose();
    }

    void run(const Eigen::Ref<const MatriXdR> &X)
    {
        const unsigned int r = X.rows();
        const unsigned int c = X.cols();
        const double mu = r * c / (4. * X.cwiseAbs().sum());
        const double lambda = 1. / (r ? r > c : c);
        const double threshold = this->thresholdScale * getFrobeniusNorm(X);
        unsigned int count = 0;

        // initialise L, S, Y zero matrices
        L = MatriXdR::Zero(r, c);
        S = MatriXdR::Zero(r, c);
        Y = MatriXdR::Zero(r, c);

        while (
            (count < this->maxCount) && (getFrobeniusNorm(X - L - S) > threshold))
        {
            const MatriXdR XS = X - S - (1 / mu) * Y;
            const MatriXdR XL = X - L + (1 / mu) * Y;
            L = truncatedSVD(XS, 1. / mu);
            S = shrinkMatrix(XL, lambda / mu);
            Y = Y + mu * (X - L - S);
            count++;
        }
    }
};
#endif // RPCA_HPP

