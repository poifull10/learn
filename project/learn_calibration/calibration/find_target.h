#pragma once

#include <opencv2/opencv.hpp>

namespace calib
{
class TargetFinder
{
public:
  TargetFinder();

  void operator()(const cv::Mat& img);

private:
};
} // namespace calib
