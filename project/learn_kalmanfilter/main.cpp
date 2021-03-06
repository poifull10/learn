
#include <iostream>

#include "generateData.h"
#include "kalmanFilter.h"

int main()
{
  float dt = 1;
  size_t N = 20;
  float sigma = 0.1;
  const Eigen::MatrixXf gt = kf::generateData(N, dt);
  const Eigen::MatrixXf observation = gt.topRows(3) + kf::noise(3, N, sigma);

  kf::KalmanFilter kf(dt);
  Eigen::VectorXf x0 = Eigen::VectorXf::Zero(6);
  Eigen::MatrixXf Sigma0 = Eigen::MatrixXf::Identity(6, 6);
  Eigen::MatrixXf SigmaV = Eigen::MatrixXf::Identity(3, 3) * sigma;
  Eigen::MatrixXf SigmaW = Eigen::MatrixXf::Identity(6, 6) * sigma;

  kf.init(x0, Sigma0, SigmaW, SigmaV);

  std::cout << "estimating ..." << std::endl;
  std::cout << kf.x().transpose() << std::endl;

  for (size_t i = 0; i < N; i++)
  {
    const auto [x, Sigma] = kf.estimate(observation.col(i));
    std::cout << "Estimated: " << x.transpose() << std::endl;
    if (i < N - 1)
      std::cout << "GT: " << gt.col(i + 1).transpose() << std::endl;
  }

  return 0;
}
