#pragma once

#include <torch/torch.h>

class SeqDiscriminator {
  public:
    SeqDiscriminator() = delete;
    explicit SeqDiscriminator(double negative_slope_value);

    torch::nn::Sequential & GetDiscriminator();

  private:
    torch::nn::Sequential discriminator;
};
