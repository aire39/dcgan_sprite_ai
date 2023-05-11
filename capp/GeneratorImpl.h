#pragma once

#include <cstdint>
#include <array>
#include <torch/torch.h>

class GeneratorImpl : public torch::nn::Module {
  public:
    GeneratorImpl(int64_t noise_size);
    torch::Tensor forward(torch::Tensor x);

  private:
    std::array<torch::nn::ConvTranspose2d, 4> convolutions;
    std::array<torch::nn::BatchNorm2d, 3> batchNormalizations;
};
