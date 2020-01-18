#include "model.h"

namespace kf
{
Eigen::VectorXf ConstantVelocityModel::operator()(
  const Eigen::VectorXf& x) const
{
  return F_ * x;
}

void ConstantVelocityModel::init()
{
  F_ = Eigen::MatrixXf::Identity(6, 6);
  F_(0, 3) = dt_;
  F_(1, 4) = dt_;
  F_(2, 5) = dt_;

  G_ = Eigen::MatrixXf::Identity(6, 6);
  H_ = Eigen::MatrixXf::Zero(3, 6);
  H_(0, 0) = 1;
  H_(1, 1) = 1;
  H_(2, 2) = 1;
}
} // namespace kf
