#include "generateData.h"

#include <random>

namespace kf
{

Eigen::MatrixXf generateData(size_t N)
{
  return Eigen::MatrixXf::Zero(3, N);
}

Eigen::MatrixXf noise(float row, float col, float stdv)
{
  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());
  std::normal_distribution dist(0.0F, stdv);
  Eigen::MatrixXf noise = Eigen::MatrixXf::Zero(row, col);

  for (size_t iy = 0; iy < row; iy++)
  {
    for (size_t ix = 0; ix < col; ix++)
    {
      noise(iy, ix) = dist(engine);
    }
  }
  return noise;
}

Eigen::MatrixXf ConstantVelocityModel::operator()(const Eigen::MatrixXf& x)
{
  assert(vel.rows() == x.rows());
  Eigen::MatrixXf y = x;
  for (size_t iw = 0; iw < x.cols(); iw++)
  {
    y.col(iw) += vel;
  }
  return y;
}

} // namespace kf
