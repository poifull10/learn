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

cv::Point2f model(float R, float u, float v, float theta)
{
  auto x = R * std::cos(theta) + u;
  auto y = R * std::sin(theta) + v;
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
  CircleEstimator(const std::vector<cv::Point2f>& points)
    : observedPoints(points), N(points.size())
  {
  }

  void setInitialGuess(float r_, float u_, float v_,
                       const std::vector<float>& thetas_)
  {
    r = r_;
    u = u_;
    v = v_;
    thetas = thetas_;
  }

  float energy() const
  {
    cv::Point2f Err;
    for (size_t i = 0; i < N; i++)
    {
      const auto sub = observedPoints[i] - model(r, u, v, thetas[i]);
      Err.x += sub.x * sub.x;
      Err.y += sub.y * sub.y;
    }
    return Err.x * Err.x + Err.y * Err.y;
  }

  cv::Mat residual() const
  {
    cv::Mat e = cv::Mat::zeros(cv::Size(1, 2 * N), CV_32F);
    for (size_t i = 0; i < N; i++)
    {
      const auto sub = observedPoints[i] - model(r, u, v, thetas[i]);
      e.at<float>(cv::Point(0, 2 * i)) = sub.x;
      e.at<float>(cv::Point(0, 2 * i + 1)) = sub.y;
    }
    return e;
  }

  void LM(size_t iter, float lambda_ = 1e-10, float tolelance = 1e-6)
  {
    lambda = lambda_;
    cv::Mat I = cv::Mat::eye(N + 3, N + 3, CV_32F);
    for (size_t i = 0; i < iter; i++)
    {
      const auto J = jac();
      const auto H = J.t() * J;
      const auto e = residual();
      cv::Mat b = J.t() * e;
      cv::Mat dx = -(H + lambda * I).inv() * b;
      float beforeEnergy = energy();
      if (cv::norm(dx) < tolelance)
      {
        return;
      }
      update(dx);
      float afterEnergy = energy();
      std::cout << "Before Energy: " << beforeEnergy << std::endl;
      std::cout << "After Energy: " << afterEnergy << std::endl;
      if (afterEnergy < beforeEnergy)
      {
        lambda *= 0.1;
        showParameters();
      }
      else
      {
        lambda *= 10;
        update(-dx);
        i--;
      }
    }
  }

  void update(const cv::Mat& dx)
  {
    dr = dx.at<float>(cv::Point(0, 0));
    du = dx.at<float>(cv::Point(1, 0));
    dv = dx.at<float>(cv::Point(2, 0));
    r += dr;
    u += du;
    v += dv;
    for (size_t i = 0; i < N; i++)
    {
      thetas[i] += dx.at<float>(cv::Point(3 + i, 0));
    }
  }

  void showParameters() const
  {
    std::cout << "==========" << std::endl;
    const auto [R, U, V] = getParams();
    std::cout << "R = " << R << std::endl;
    std::cout << "u = " << U << std::endl;
    std::cout << "v = " << V << std::endl;
    std::cout << "dr = " << dr << std::endl;
    std::cout << "du = " << du << std::endl;
    std::cout << "dv = " << dv << std::endl;
    std::cout << "E = " << energy() << std::endl;
  }

  cv::Mat jac() const
  {
    cv::Mat J = cv::Mat::zeros(2 * N, N + 3, CV_32F);
    for (size_t i = 0; i < N; i++)
    {
      J.at<float>(cv::Point(0, 2 * i)) = -std::cos(thetas[i]);
      J.at<float>(cv::Point(0, 2 * i + 1)) = -std::sin(thetas[i]);
      J.at<float>(cv::Point(1, 2 * i)) = -1;
      J.at<float>(cv::Point(2, 2 * i + 1)) = -1;
      J.at<float>(cv::Point(3 + i, 2 * i)) = r * std::sin(thetas[i]);
      J.at<float>(cv::Point(3 + i, 2 * i + 1)) = -r * std::cos(thetas[i]);
    }
    return J;
  }

  std::tuple<float, float, float> getParams() const { return {r, u, v}; }

private:
  std::vector<cv::Point2f> observedPoints;
  float lambda;
  std::vector<float> thetas;
  size_t N;
  float r;
  float u;
  float v;

  float dr, du, dv;
};

int main()
{
  const float n = 100;
  const float r = 300;
  const float u = 512;
  const float v = 512;

  // data generation
  const auto [gtPoints, gtThetas] = generateGT(n, r, u, v);
  const auto [observedPoints, observedThetas] = addNoise(gtPoints, gtThetas);

  CircleEstimator ce(observedPoints);
  auto initR = 10;
  auto initU = 30;
  auto initV = 10;
  ce.setInitialGuess(initR, initU, initV, observedThetas);

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
