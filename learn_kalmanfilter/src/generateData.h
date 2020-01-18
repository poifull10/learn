#pragma once

#include <Eigen/Core>

namespace kf
{

Eigen::MatrixXf generateData(size_t N);
Eigen::MatrixXf noise(float col, float row, float stdv);
Eigen::MatrixXf addNoise(const Eigen::MatrixXf& value, float stdv);

class Model
{
public:
  Model() {}

  virtual Eigen::MatrixXf operator()(const Eigen::MatrixXf& x) = 0;

private:
  Eigen::MatrixXf A_;
};
} // namespace kf