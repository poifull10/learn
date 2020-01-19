#pragma once

#include <Eigen/Core>

namespace kf
{

Eigen::MatrixXf generateData(size_t N, float dt);
Eigen::MatrixXf noise(float col, float row, float stdv);

} // namespace kf
