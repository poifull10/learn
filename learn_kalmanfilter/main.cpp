
#include "generateData.h"

#include <iostream>

int main()
{
  std::cout << kf::generateData(10) << std::endl;
  std::cout << kf::generateData(10) + kf::noise(3, 10, 1) << std::endl;
  return 0;
}
