#include <kalmanFilter.h>

#include <Eigen/LU>
#include <iostream>

namespace kf
{
std::pair<Eigen::VectorXf, Eigen::MatrixXf> KalmanFilter::estimate(
  const Eigen::VectorXf& y_)
{
  assert(isInitialized_);
  Eigen::MatrixXf K =
    Sigma_ * model_->H().transpose() *
    (model_->H() * Sigma_ * model_->H().transpose() + SigmaV_).inverse();
  Eigen::VectorXf xk_k = x_ + K * (y_ - model_->H() * x_);
  Eigen::MatrixXf Sigmak_k = Sigma_ - K * model_->H() * Sigma_;

  x_ = model_->F() * xk_k;
  Sigma_ = model_->F() * Sigmak_k * model_->F().transpose() +
           model_->G() * SigmaW_ * model_->G().transpose();
  return {x_, Sigma_};
}

} // namespace kf
