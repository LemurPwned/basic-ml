#include <gtest/gtest.h>
#include "../python/rpca.hpp"

TEST(RPCATest, MainTest)
{
    const auto X = Eigen::MatrixXd::Random(100, 100);
    RPCA rpca(100);
    rpca.run(X);
    // std::cout << rpca.getL() << std::endl;
    // std::cout << rpca.getS() << std::endl;
}