
#include <iostream>

#include "generateData.h"
#include "kalmanFilter.h"

int main()
{
  float dt = 1;
  size_t N = 20;
  float sigma = 0.1;
  const Eigen::MatrixXf gt = kf::generateData(N, dt);
  const Eigen::MatrixXf observation = gt + kf::noise(6, N, sigma);

  kf::KalmanFilter kf(dt);
  Eigen::VectorXf x0 = gt.col(0);
  Eigen::MatrixXf Sigma0 = Eigen::MatrixXf::Identity(6, 6);
  Eigen::MatrixXf SigmaV = Eigen::MatrixXf::Identity(6, 6) * sigma;
  Eigen::MatrixXf SigmaW = Eigen::MatrixXf::Identity(6, 6) * sigma;

  kf.init(x0, Sigma0, SigmaV, SigmaW);

  std::cout << "estimating ..." << std::endl;

  for (size_t i = 1; i < N; i++)
  {
    const auto [x, Sigma] = kf.estimate(observation.col(i));
    std::cout << "Estimated: " << x.transpose() << std::endl;
    std::cout << "GT: " << gt.col(i).transpose() << std::endl;
  }

  return 0;
}
