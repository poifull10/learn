
#include "gtest/gtest.h"
#include "model.h"

TEST(ConstantVelocityModel, test_operator)
{
  float dt = 0.2;
  kf::ConstantVelocityModel cvm(dt);
  Eigen::VectorXf vel = Eigen::VectorXf::Zero(6);
  vel(3) = 1.0F;
  vel(4) = 2.0F;
  vel(5) = 3.3F;

  Eigen::MatrixXf v = vel;
  v = cvm(v);

  EXPECT_FLOAT_EQ(vel(3) * dt, v(0));
  EXPECT_FLOAT_EQ(vel(4) * dt, v(1));
  EXPECT_FLOAT_EQ(vel(5) * dt, v(2));
  EXPECT_FLOAT_EQ(vel(3), v(3));
  EXPECT_FLOAT_EQ(vel(4), v(4));
  EXPECT_FLOAT_EQ(vel(5), v(5));

  v = cvm(v);

  EXPECT_FLOAT_EQ(vel(3) * 2 * dt, v(0));
  EXPECT_FLOAT_EQ(vel(4) * 2 * dt, v(1));
  EXPECT_FLOAT_EQ(vel(5) * 2 * dt, v(2));
  EXPECT_FLOAT_EQ(vel(3), v(3));
  EXPECT_FLOAT_EQ(vel(4), v(4));
  EXPECT_FLOAT_EQ(vel(5), v(5));
}
