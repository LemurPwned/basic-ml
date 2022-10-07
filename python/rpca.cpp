#include <stdio.h>
#include <Eigen/Dense>

class RPCA
{

private:
    double thresh = 0.1;
    unsigned int maxCount = 1000;
    double mu = 0.1 

public:
    RPCA()
    {
    }

    double getFrobeniusNorm(Eigen::MatrixXd X)
    {
        return 0;
    }

    void run(Eigen::MatrixXd X)
    {
        // X
        unsigned int count = 0;
        while (
            getFrobeniusNorm(X) < this->thres)
            &&(count < this->maxCount)
            {

                count += 1;
            }
    }
};