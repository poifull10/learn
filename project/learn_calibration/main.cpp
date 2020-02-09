#include <opencv2/opencv.hpp>
#include <string>

#include "target_detector.h"

int main(int argc, char** argv)
{
  cv::Mat img = cv::imread(std::string(argv[1]) + "/002.png");
  // cal::ChessBoardTargetDetector td(4, 4);
  // auto output = td.extract(img);
  // cv::drawChessboardCorners(img, cv::Size(4, 4), output, true);

  // cv::imshow("show", img);
  // cv::waitKey(0);
  std::vector<cv::Point2f> corners;
  cv::findChessboardCorners(img, cv::Size(4, 4), corners);
  std::vector<std::vector<cv::Point3f>> objp = {{}};

  for (size_t ih = 0; ih < 4; ih++)
  {
    for (size_t iw = 0; iw < 4; iw++)
    {
      cv::Point3f p;
      p.x = -0.6 + 0.4 * iw;
      p.y = 0.6 - 0.4 * ih;
      p.z = 0;
      objp[0].push_back(p);
    }
  }

  for (const auto& p : objp[0])
  {
    std::cout << p << std::endl;
  }

  cv::Mat K;
  cv::Mat r, t;
  cv::Mat dist = cv::Mat::zeros(cv::Size(4, 1), CV_32F);
  cv::Mat stdDeviationsInt, stdDeviationsExt;
  cv::Mat perViewErrors;
  std::vector<std::vector<cv::Point2f>> cor = {corners};
  cv::calibrateCamera(objp, cor, cv::Size(1024, 1024), K, dist, r, t,
                      stdDeviationsInt, stdDeviationsExt, perViewErrors);
  std::cout << K << std::endl;
  std::cout << r << std::endl;
  std::cout << t << std::endl;

  cv::drawFrameAxes(img, K, dist, r, t, 1.0F, 1);

  cv::imshow("show", img);
  cv::waitKey(0);
  return 0;
}