#include <iostream>
#include <torch/torch.h>
#include "GeneratorImpl.h"
#include "SeqDiscriminator.h"
#include "ImageFolder.h"

#define SAVE_IMAGES_RGB true
#if SAVE_IMAGES_RGB
#include <fstream>
#endif


int main([[maybe_unused]] int argc, [[maybe_unused]] char*argv[])
{
  constexpr int64_t knoise_size = 100;
  constexpr int64_t kbatch_size = 64;
  constexpr int32_t knumber_of_workers = 4;
  constexpr int32_t knumber_of_epochs = 108;//1000;
  constexpr bool kenforce_order = false;
  constexpr double klr = 2e-4;
  constexpr double kbeta1 = 0.5;
  constexpr double kbeta2 = 0.999;
  constexpr int64_t klaten = 100;
  constexpr int64_t klog_interval = 4;
  constexpr int64_t kcheckpoint_interval = 200;
  constexpr int64_t knumber_of_samples_per_checkpoint = 64;
  constexpr char default_image_path[] = "data/creatures/images";

  torch::manual_seed(42);

  torch::Device device(torch::kCPU);

  if (torch::cuda::is_available())
  {
    device = torch::Device(torch::kCUDA);
    std::cout << "Cuda available! Will use found GPU!" << std::endl;
  }

  Generator generator (knoise_size);
  generator->to(device);

  SeqDiscriminator seq_discriminator(0.2);
  auto & discriminator = seq_discriminator.GetDiscriminator();
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

  torch::optim::Adam discriminator_optimizer(
     discriminator->parameters()
    ,torch::optim::AdamOptions(klr)
      .betas(std::make_tuple(kbeta1, kbeta2))
  );

  const int64_t batches_per_epoch = std::ceil(static_cast<double>(dataset.size().value()) / static_cast<double>(kbatch_size));
  int64_t checkpoint_counter = 1;

  for(size_t epoch=1; epoch<=knumber_of_epochs; ++epoch)
  {
    int64_t batch_index = 0;

    for(auto& batch : *data_loader)
    {
      // train discriminator

      // real images
      discriminator->zero_grad();
      torch::Tensor real_images = batch.data.to(device);
      torch::Tensor real_labels = torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);
      torch::Tensor real_output = discriminator->forward(real_images).view(-1);
      torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
      d_loss_real.backward();

      // fake images
      torch::Tensor noise = torch::randn({batch.data.size(0), klaten, 1, 1}, device);
      torch::Tensor fake_images = generator->forward(noise);
      torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);
      torch::Tensor fake_output = discriminator->forward(fake_images.detach()).view(-1);
      torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
      d_loss_fake.backward();

      torch::Tensor d_loss = d_loss_real + d_loss_fake;
      discriminator_optimizer.step();

      // train generator
      generator->zero_grad();

      fake_labels.fill_(1);
      fake_output = discriminator->forward(fake_images).view(-1);

      torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
      g_loss.backward();

      generator_optimizer.step();

      batch_index++;
      checkpoint_counter++;

      if ((batch_index % klog_interval) == 0)
      {
        std::cout << "[" << epoch << "/" << knumber_of_epochs << "] [" << batch_index << "/" << batches_per_epoch << "] D_loss: " << d_loss.item<float>() << " G_loss: " << g_loss.item<float>() << " -- batch_sizes: " << batch.data.sizes() << "\n";
      }

      if ((checkpoint_counter % kcheckpoint_interval) == 0)
      {
        torch::save(generator, "gen_chkpt.pt");
        torch::save(generator_optimizer, "genop_chkpt.pt");
        torch::save(discriminator, "dis_chkpt.pt");
        torch::save(generator_optimizer, "disop_chkpt.pt");

        torch::Tensor samples = generator->forward(torch::randn({knumber_of_samples_per_checkpoint, klaten, 1, 1}, device));
        torch::save(((samples + 1) / 2.0), torch::str("dcgan-sample-", checkpoint_counter, ".pt"));

#if SAVE_IMAGES_RGB
        torch::Tensor rgb = generator->forward(torch::randn({knumber_of_samples_per_checkpoint, klaten, 1, 1}, device));
        std::cout << "rgb_tensor w: " << rgb.size(3) << " h: " << rgb.size(2) << " c: " << rgb.size(1) << " layers:" << rgb.size(0) << " dim: " << rgb.dim() << std::endl;

        constexpr int32_t rgb_padding = 0;
        constexpr int32_t rgb_pad_value = 0;
        int32_t nmaps = rgb.size(0);
        int32_t xmaps = 8;
        auto ymaps = static_cast<int32_t>(std::ceil(static_cast<float>((nmaps) / static_cast<float>(xmaps))));
        int32_t height = rgb.size(2) + rgb_padding;
        int32_t width = rgb.size(3) + rgb_padding;
        int32_t nchannels = rgb.size(1);

        int32_t k=0;
        auto grid = rgb.new_full({nchannels, (height*ymaps + rgb_padding), (width*xmaps + rgb_padding)}, rgb_pad_value);
        for (int32_t y=0; y<ymaps; y++)
        {
          for (int32_t x=0; x<xmaps; x++)
          {
            if (k >= nmaps)
            {
              break;
            }

            grid.narrow(1, y*height+rgb_padding, height-rgb_padding).narrow(2, x*width+rgb_padding, width-rgb_padding).copy_(rgb[k]);
            k++;
          }
        }

        grid = (grid + 1) / 2.0;
        grid = grid.squeeze().detach();
        grid = grid.permute({1,2,0}).contiguous();
        grid = grid.mul(255).clamp(0, 255).to(torch::kU8);
        grid = grid.to(torch::kCPU);

        std::cout << "grid_tensor w: " << grid.size(2) << " h: " << grid.size(1) << " c: " << grid.size(0) << " size: " << grid.sizes() << std::endl;

        const uint8_t* hh = grid.data_ptr<uint8_t>();
        int w = grid.size(2);
        int h = grid.size(1);
        int c = grid.size(0);

        std::ofstream f_out("test" + std::to_string(checkpoint_counter) + ".rgb");
        f_out.write((char*)hh, (w*h*c));
        f_out.flush();
        f_out.close();
#endif

        std::cout << "\n-> checkpoint " << checkpoint_counter << "\n\n";
      }
    }
  }

  std::cout << "training complete!" << std::endl;

  return 0;
}