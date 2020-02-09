#include <target_detector.h>

namespace cal
{
ChessBoardTargetDetector::ChessBoardTargetDetector(size_t width, size_t height)
  : w(width), h(height)
{
}

cv::Mat ChessBoardTargetDetector::extract(const cv::Mat& img)
{
  cv::Mat corners;
  cv::findChessboardCorners(img, cv::Size(w, h), corners);
  return corners;
}

}; // namespace cal
