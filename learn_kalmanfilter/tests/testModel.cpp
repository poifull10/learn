#include "generateData.h"
#include "gtest/gtest.h"

TEST(ConstantVelocityModel, test_operator)
{
  kf::ConstantVelocityModel cvm(3);
  Eigen::MatrixXf vel = cvm.getVelocity();

  Eigen::MatrixXf v = Eigen::MatrixXf::Zero(3, 10);
  v = cvm(v);

  for (size_t ih = 0; ih < v.rows(); ih++)
  {
    for (size_t iw = 0; iw < v.cols(); iw++)
    {
      EXPECT_FLOAT_EQ(vel(ih, 0), v(ih, iw));
    }
  }

  v = cvm(v);

  for (size_t ih = 0; ih < v.rows(); ih++)
  {
    for (size_t iw = 0; iw < v.cols(); iw++)
    {
      EXPECT_FLOAT_EQ(vel(ih, 0) * 2, v(ih, iw));
    }
  }

  EXPECT_TRUE(true);
}
