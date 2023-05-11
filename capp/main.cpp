#include <iostream>
#include <memory>
#include <torch/torch.h>
#include "GeneratorImpl.h"
#include "DiscriminatorImpl.h"
#include "ImageFolder.h"

#include <vector>
#include <filesystem>

int main(int argc, char*argv[])
{
  constexpr int64_t noise_size = 100;

  auto generator = std::make_unique<GeneratorImpl>(noise_size);
  auto discriminator = std::make_unique<DiscriminatorImpl>();

  auto dataset = ImageFolder("data/creatures/images", '_');

  return 0;
}