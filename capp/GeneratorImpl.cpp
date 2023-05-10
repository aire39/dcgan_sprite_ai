#include "GeneratorImpl.h"

GeneratorImpl::GeneratorImpl(int64_t noise_size)
  : convolutions{{
      {torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(noise_size, 256, 4).bias(false))}
     ,{torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(256, 128, 4).stride(2).padding(1).bias(false))}
     ,{torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1).bias(false))}
     ,{torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 1, 4).stride(2).padding(1).bias(false))}
  }}
  , batchNormalizations{{
     {torch::nn::BatchNorm2d(256)}
    ,{torch::nn::BatchNorm2d(128)}
    ,{torch::nn::BatchNorm2d(64)}
  }}
{
  for (size_t i=0; i<convolutions.size(); i++)
  {
    this->register_module("conv" + std::to_string(i), convolutions[i]);
  }

  for (size_t i=0; i<batchNormalizations.size(); i++)
  {
    this->register_module("batch_norm" + std::to_string(i), batchNormalizations[i]);
  }
}

torch::Tensor GeneratorImpl::forward(torch::Tensor x)
{
  for (size_t i=0; i<convolutions.size()-1; i++)
  {
    x = torch::relu(batchNormalizations[i](convolutions[i](x)));
  }

  x = torch::tanh(convolutions[convolutions.size()-1](x));

  return x;
}