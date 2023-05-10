#include <iostream>
#include <memory>
#include "GeneratorImpl.h"
#include "DiscriminatorImpl.h"

int main(int argc, char*argv[])
{
  constexpr int64_t noise_size = 100;

  auto generator = std::make_unique<GeneratorImpl>(noise_size);
  auto discriminator = std::make_unique<DiscriminatorImpl>();

  std::cout << "Hello World!" << std::endl;

  return 0;
}