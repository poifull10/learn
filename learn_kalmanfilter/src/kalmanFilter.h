#pragma once

#include <Eigen/Core>
#include <memory>
#include <utility>

#include "model.h"

namespace kf
{
class KalmanFilter
{
public:
  KalmanFilter(float dt)
    : model_(std::make_unique<ConstantVelocityModel>(dt)), isInitialized_(false)
  {
  }

  void init(const Eigen::VectorXf& x0, const Eigen::MatrixXf& Sigma0,
            const Eigen::MatrixXf& noiseOfXCov,
            const Eigen::MatrixXf& noiseOfYCov)
  {
    x_ = x0;
    Sigma_ = Sigma0;
    SigmaW_ = noiseOfXCov;
    SigmaV_ = noiseOfYCov;
    isInitialized_ = true;
  }

  std::pair<Eigen::VectorXf, Eigen::MatrixXf> estimate(
    const Eigen::VectorXf& y);

  Eigen::VectorXf x() const { return x_; }

private:
  std::unique_ptr<Model> model_;
  Eigen::MatrixXf K_;
  Eigen::VectorXf x_;
  Eigen::MatrixXf Sigma_, SigmaW_, SigmaV_;
  bool isInitialized_;
};

} // namespace kf
