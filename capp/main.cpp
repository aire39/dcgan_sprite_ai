#include <iostream>
#include <memory>
#include <torch/torch.h>
#include "GeneratorImpl.h"
#include "DiscriminatorImpl.h"
#include "ImageFolder.h"

#include <vector>
#include <filesystem>

TORCH_MODULE(Generator);
TORCH_MODULE(Discriminator);

int main(int argc, char*argv[])
{
  constexpr int64_t knoise_size = 100;
  constexpr int64_t kbatch_size = 64;
  constexpr int32_t knumber_of_workers = 4;
  constexpr bool kenforce_order = false;
  constexpr double klr = 0.002;
  constexpr double kbeta1 = 0.002;
  constexpr double kbeta2 = 0.002;
  constexpr char default_image_path[] = "data/creatures/images";

  torch::Device device(torch::kCPU);

  if (torch::cuda::is_available())
  {
    device = torch::Device(torch::kCUDA);
  }

  Generator generator (knoise_size);
  generator->to(device);

  Discriminator discriminator;
  discriminator->to(device);

  auto dataset = ImageFolder(default_image_path, '_')
          .map(torch::data::transforms::Normalize<>(0.5, 0.5))
          .map(torch::data::transforms::Stack<>());

  auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
          dataset
         ,torch::data::DataLoaderOptions().workers(knumber_of_workers)
           .batch_size(kbatch_size)
           .enforce_ordering(kenforce_order)
         );

  torch::optim::Adam generator_optimizer(
     generator->parameters()
    ,torch::optim::AdamOptions(klr)
      .betas(std::make_tuple(kbeta1, kbeta2))
  );

  const int64_t batches_per_epoch = std::ceil(dataset.size().value() / static_cast<double>(kbatch_size));

  return 0;
}