#pragma once

#include <cstdint>
#include <array>
#include <torch/torch.h>

class DiscriminatorImpl : public torch::nn::Module {
  public:
    DiscriminatorImpl();
    torch::Tensor forward(torch::Tensor x);

  private:
    std::array<torch::nn::Conv2d, 4> convolutions;
    std::array<torch::nn::BatchNorm2d, 2> batchNormalizations;
};

TORCH_MODULE(Discriminator);