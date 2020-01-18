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

  virtual Eigen::VectorXf operator()(const Eigen::VectorXf& x) const = 0;

private:
  Eigen::MatrixXf A_;
};

class ConstantVelocityModel : public Model
{
public:
  ConstantVelocityModel(float dt) : dt_(dt) { init(); }

  Eigen::VectorXf operator()(const Eigen::VectorXf& x) const override;
  Eigen::MatrixXf F() const { return F_; }

private:
  void init();
  Eigen::MatrixXf F_;
  float dt_;
};
} // namespace kf
