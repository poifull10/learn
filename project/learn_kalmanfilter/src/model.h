#pragma once

#include <Eigen/Core>

namespace kf
{

class Model
{
public:
  Model() {}
  virtual ~Model() {}
  virtual void init() = 0;
  virtual Eigen::VectorXf operator()(const Eigen::VectorXf& x) const = 0;
  Eigen::MatrixXf F() const { return F_; }
  Eigen::MatrixXf G() const { return G_; }
  Eigen::MatrixXf H() const { return H_; }

protected:
  Eigen::MatrixXf F_, G_, H_;
};

class ConstantVelocityModel : public Model
{
public:
  ConstantVelocityModel(float dt) : dt_(dt) { init(); }
  ~ConstantVelocityModel() {}
  Eigen::VectorXf operator()(const Eigen::VectorXf& x) const override;

  void init() override;

private:
  float dt_;
};

} // namespace kf
