#pragma once

#include <Eigen/Core>

namespace kf
{

Eigen::MatrixXf generateData(size_t N);
Eigen::MatrixXf noise(float col, float row, float stdv);

class Model
{
public:
  Model() {}

  virtual Eigen::MatrixXf operator()(const Eigen::MatrixXf& x) = 0;

private:
  Eigen::MatrixXf A_;
};

class ConstantVelocityModel : public Model
{
public:
  ConstantVelocityModel(size_t row) : vel(Eigen::MatrixXf::Random(row, 1)) {}
  ConstantVelocityModel(const Eigen::MatrixXf& velocity) : vel(velocity) {}

  Eigen::MatrixXf getVelocity() const { return vel; }
  Eigen::MatrixXf operator()(const Eigen::MatrixXf& x) override;

private:
  Eigen::MatrixXf vel;
};
} // namespace kf
