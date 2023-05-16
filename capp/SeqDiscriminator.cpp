//
// Created by Sean on 5/11/2023.
//

#include "SeqDiscriminator.h"

SeqDiscriminator::SeqDiscriminator(double negative_slope_value)
{
  discriminator = torch::nn::Sequential(
     torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 4).stride(2).padding(1).bias(false))
    ,torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(negative_slope_value).inplace(true))

    ,torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false))
    ,torch::nn::BatchNorm2d(128)
    ,torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(negative_slope_value).inplace(true))

    ,torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false))
    ,torch::nn::BatchNorm2d(256)
    ,torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(negative_slope_value).inplace(true))

    ,torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 4).stride(2).padding(1).bias(false))
    ,torch::nn::BatchNorm2d(512)
    ,torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(negative_slope_value).inplace(true))

    ,torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 1, 4).stride(1).padding(0).bias(false))
    ,torch::nn::Sigmoid()
  );
}

torch::nn::Sequential & SeqDiscriminator::GetDiscriminator()
{
  return discriminator;
}