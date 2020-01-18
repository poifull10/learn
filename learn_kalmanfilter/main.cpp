
#include "generateData.h"

#include <iostream>

int main()
{
  std::cout << kf::generateData(10) << std::endl;
  std::cout << kf::addNoise(kf::generateData(10), 1.0) << std::endl;
  return 0;
}
