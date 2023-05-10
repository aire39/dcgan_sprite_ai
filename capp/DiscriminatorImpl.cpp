#include "DiscriminatorImpl.h"

DiscriminatorImpl::DiscriminatorImpl()
  : convolutions{{
      {torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 3).bias(false))}
     ,{torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(2).padding(1).bias(false))}
     ,{torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(2).padding(1).bias(false))}
     ,{torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 1, 3).stride(2).padding(1).bias(false))}
  }}
  , batchNormalizations{{
     {torch::nn::BatchNorm2d(128)}
    ,{torch::nn::BatchNorm2d(256)}
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

torch::Tensor DiscriminatorImpl::forward(torch::Tensor x)
{
  for (size_t i=0; i<convolutions.size()-1; i++)
  {
    x = torch::leaky_relu(batchNormalizations[i](convolutions[i](x)), 0.2);
  }

  x = torch::sigmoid(convolutions[convolutions.size()-1](x));

  return x;
}
