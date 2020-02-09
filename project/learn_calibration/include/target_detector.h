#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace cal
{

class ChessBoardTargetDetector
{
public:
  ChessBoardTargetDetector(size_t width, size_t height);
  cv::Mat extract(const cv::Mat& img);

private:
  size_t w, h;
};

}; // namespace cal
