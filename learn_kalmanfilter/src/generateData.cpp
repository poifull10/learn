
#include "generateData.h"

#include <iostream>
#include <random>

namespace kf
{

Eigen::MatrixXf generateData(size_t N, float dt)
{
  Eigen::MatrixXf points = Eigen::MatrixXf::Zero(6, N);
  float vx = 5;
  float vy = 3;
  float vz = -4;

  for (size_t i = 0; i < N; i++)
  {
    points(0, i) = i * dt * vx;
    points(1, i) = i * dt * vy;
    points(2, i) = i * dt * vz;
    points(3, i) = vx;
    points(4, i) = vy;
    points(5, i) = vz;
  }
  return points;
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

} // namespace kf
