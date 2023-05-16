#include "GeneratorImpl.h"

GeneratorImpl::GeneratorImpl(int64_t noise_size)
  : convolutions{{
      {torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(noise_size, 512, 4).stride(1).padding(0).bias(false))}
     ,{torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(512, 256, 4).stride(2).padding(1).bias(false))}
     ,{torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(256, 128, 4).stride(2).padding(1).bias(false))}
     ,{torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1).bias(false))}
     ,{torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 3, 4).stride(2).padding(1).bias(false))}
  }}
  , batchNormalizations{{
     {torch::nn::BatchNorm2d(512)}
    ,{torch::nn::BatchNorm2d(256)}
    ,{torch::nn::BatchNorm2d(128)}
    ,{torch::nn::BatchNorm2d(64)}
  }}
{
  for (size_t i=0; i<convolutions.size(); i++)
  {
    this->register_module("conv" + std::to_string(i+1), convolutions[i]);
  }

  for (size_t i=0; i<batchNormalizations.size(); i++)
  {
    this->register_module("batch_norm" + std::to_string(i+1), batchNormalizations[i]);
  }
}

torch::Tensor GeneratorImpl::forward(torch::Tensor x)
{
  for (size_t i=0; i<batchNormalizations.size(); i++)
  {
    x = torch::nn::functional::relu(batchNormalizations[i](convolutions[i](x)), torch::nn::functional::ReLUFuncOptions().inplace(true));
  }

  x = torch::tanh(convolutions[convolutions.size()-1](x));

  return x;
}
