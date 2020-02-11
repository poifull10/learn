#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

cv::Point2f noise(float stdv = 1.0F)
{
  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());
  std::normal_distribution dist(0.0F, stdv);
  return cv::Point2f(dist(engine), dist(engine));
}

cv::Point2f model(float r, float u, float v, float theta)
{
  const auto x = r * std::cos(theta) + u;
  const auto y = r * std::sin(theta) + v;
  return cv::Point2f(x, y);
}

std::pair<std::vector<cv::Point2f>, std::vector<float>> generateGT(size_t N,
                                                                   float R,
                                                                   float u,
                                                                   float v)
{
  std::vector<cv::Point2f> ret;
  std::vector<float> thetas;
  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());
  std::uniform_real_distribution dist(0.0, 3.141592653 * 2);
  for (size_t i = 0; i < N; i++)
  {
    const auto rndv = dist(engine);
    thetas.push_back(static_cast<float>(rndv));
    ret.push_back(model(R, u, v, rndv));
  }
  return {ret, thetas};
}

std::pair<std::vector<cv::Point2f>, std::vector<float>> addNoise(
  const std::vector<cv::Point2f>& gtPoints, const std::vector<float>& gtThetas)
{
  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());
  std::normal_distribution dist(0.0F, 0.2F);

  std::vector<cv::Point2f> observedPoints;
  std::vector<float> observedThetas;
  for (size_t i = 0; i < gtPoints.size(); i++)
  {
    const auto noisyP = gtPoints[i] + noise(10);
    const auto noisyTheta = gtThetas[i] + dist(engine);
    observedPoints.push_back(noisyP);
    observedThetas.push_back(noisyTheta);
  }
  return {observedPoints, observedThetas};
}

class CircleEstimator
{
public:
  CircleEstimator(const std::vector<cv::Point2f>& points, float r, float u,
                  float v, const std::vector<float>& thetas)
    : observedPoints_(points)
    , N_(points.size())
    , r_(r)
    , u_(u)
    , v_(v)
    , thetas_(thetas)
  {
  }

private:
  std::vector<cv::Point2f> observedPoints_;
  size_t N_;
  float r_, u_, v_;
  std::vector<float> thetas_;

  float dr_, du_, dv_;

public:
  float energy() const
  {
    const auto e = residual(observedPoints_);
    return cv::Mat((e.t() * e)).at<float>(cv::Point(0, 0));
  }

  cv::Mat residual(const std::vector<cv::Point2f> points) const
  {
    cv::Mat e = cv::Mat::zeros(cv::Size(1, 2 * N_), CV_32F);
    for (size_t i = 0; i < N_; i++)
    {
      const auto sub = points[i] - model(r_, u_, v_, thetas_[i]);
      e.at<float>(cv::Point(0, 2 * i)) = sub.x;
      e.at<float>(cv::Point(0, 2 * i + 1)) = sub.y;
    }
    return e;
  }

  void LM(size_t iter, float lambda = 1e-10, float tolelance = 1e-6)
  {
    auto lambda_ = lambda;
    auto I = cv::Mat::eye(N_ + 3, N_ + 3, CV_32F);
    for (size_t i = 0; i < iter; i++)
    {
      const auto J = jac();
      const auto e = residual(observedPoints_);
      const auto b = J.t() * e;
      const auto dx = -(J.t() * J + lambda_ * I).inv() * b;
      const auto beforeEnergy = energy();
      if (cv::norm(dx) < tolelance)
      {
        return;
      }
      update(dx);
      const auto afterEnergy = energy();
      std::cout << "Before Energy: " << beforeEnergy << std::endl;
      std::cout << "After Energy: " << afterEnergy << std::endl;
      if (afterEnergy < beforeEnergy)
      {
        lambda_ *= 0.1;
        showParameters();
      }
      else
      {
        lambda_ *= 10;
        update(-dx);
        i--;
      }
    }
  }

  void update(const cv::Mat& dx)
  {
    dr_ = dx.at<float>(cv::Point(0, 0));
    du_ = dx.at<float>(cv::Point(1, 0));
    dv_ = dx.at<float>(cv::Point(2, 0));
    r_ += dr_;
    u_ += du_;
    v_ += dv_;
    for (size_t i = 0; i < N_; i++)
    {
      thetas_[i] += dx.at<float>(cv::Point(3 + i, 0));
    }
  }

  void showParameters() const
  {
    std::cout << "==========" << std::endl;
    const auto [R, U, V] = getParams();
    std::cout << "R = " << R << std::endl;
    std::cout << "u = " << U << std::endl;
    std::cout << "v = " << V << std::endl;
    std::cout << "dr = " << dr_ << std::endl;
    std::cout << "du = " << du_ << std::endl;
    std::cout << "dv = " << dv_ << std::endl;
    std::cout << "E = " << energy() << std::endl;
  }

  cv::Mat jac() const
  {
    cv::Mat J = cv::Mat::zeros(2 * N_, N_ + 3, CV_32F);
    for (size_t i = 0; i < N_; i++)
    {
      J.at<float>(cv::Point(0, 2 * i)) = -std::cos(thetas_[i]);
      J.at<float>(cv::Point(0, 2 * i + 1)) = -std::sin(thetas_[i]);
      J.at<float>(cv::Point(1, 2 * i)) = -1;
      J.at<float>(cv::Point(2, 2 * i + 1)) = -1;
      J.at<float>(cv::Point(3 + i, 2 * i)) = r_ * std::sin(thetas_[i]);
      J.at<float>(cv::Point(3 + i, 2 * i + 1)) = -r_ * std::cos(thetas_[i]);
    }
    return J;
  }

  std::tuple<float, float, float> getParams() const { return {r_, u_, v_}; }
};

int main()
{
  const float n = 300;
  const float r = 300;
  const float u = 512;
  const float v = 512;

  // data generation
  const auto [gtPoints, gtThetas] = generateGT(n, r, u, v);
  const auto [observedPoints, observedThetas] = addNoise(gtPoints, gtThetas);

  auto initR = 10;
  auto initU = 30;
  auto initV = 10;
  CircleEstimator ce(observedPoints, initR, initU, initV, observedThetas);

  std::cout << "Initial Guess is given." << std::endl;
  ce.showParameters();
  ce.LM(10);
  const auto [R, U, V] = ce.getParams();

  cv::Mat img = cv::Mat::zeros(cv::Size(u * 2, v * 2), CV_8UC3);
  for (size_t i = 0; i < gtPoints.size(); i++)
  {
    auto& vi = img.at<cv::Vec3b>(observedPoints[i]);
    vi[1] = 255;

    auto& vgt = img.at<cv::Vec3b>(gtPoints[i]);
    vgt[0] = 255;
  }
  cv::circle(img, cv::Point(initU, initV), initR, cv::Scalar(128, 128, 128));
  cv::circle(img, cv::Point(U, V), R, cv::Scalar(0, 0, 255));

  cv::imwrite("result.png", img);

  return 0;
}
