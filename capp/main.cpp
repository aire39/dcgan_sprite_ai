#include <iostream>
#include <torch/torch.h>
#include "GeneratorImpl.h"
#include "SeqDiscriminator.h"
#include "ImageFolder.h"
#include "DCGANUtils.h"
#include "Window.h"

#define SAVE_IMAGES_RGB true
#if SAVE_IMAGES_RGB
#include <fstream>
#endif


int main([[maybe_unused]] int argc, [[maybe_unused]] char*argv[])
{
  // Setup Window To See DCGAN running in real time

  Window window;
  window.Run();

  // DCGAN Setup and Training

  constexpr int32_t image_size = 64; // WxH
  constexpr int64_t knoise_size = 100;
  constexpr int64_t kbatch_size = 128;
  constexpr int32_t knumber_of_workers = 4;
  constexpr int32_t knumber_of_epochs = 1000;
  constexpr bool kenforce_order = false;
  constexpr double klr = 2e-4;
  constexpr double kbeta1 = 0.5;
  constexpr double kbeta2 = 0.999;
  constexpr int64_t klaten = 100;
  constexpr int64_t klog_interval = 4;
  constexpr int64_t kcheckpoint_interval = 200;
  constexpr int64_t kdisplay_interval = 10;
  constexpr int64_t knumber_of_samples_per_checkpoint = 64;
  constexpr char default_image_path[] = "data/creatures/images";

  torch::manual_seed(999);

  torch::Device device(torch::kCPU);

  if (torch::cuda::is_available())
  {
    device = torch::Device(torch::kCUDA);
    std::cout << "Cuda available! Will use found GPU!" << std::endl;
  }

  auto weights = torch::nn::Module::NamedModuleApplyFunction([](const std::string & name, torch::nn::Module & m){
      if (name.find("Conv2d") != std::string::npos)
      {
        auto & conv2d = reinterpret_cast<torch::nn::Conv2d&>(m);
        torch::nn::init::normal_(conv2d->weight.data(), 0.0, 0.02);
      }
      else if (name.find("ConvTranspose2d") != std::string::npos)
      {
        auto & conv2d = reinterpret_cast<torch::nn::ConvTranspose2d&>(m);
        torch::nn::init::normal_(conv2d->weight.data(), 0.0, 0.02);
      }
      else if (name.find("BatchNorm2d") != std::string::npos)
      {
        auto & batch_norm = reinterpret_cast<torch::nn::BatchNorm2d&>(m);
        torch::nn::init::normal_(batch_norm->weight.data(), 1.0, 0.02);
        torch::nn::init::constant_(batch_norm->bias.data(), 0);
      }
  });

  Generator generator (knoise_size);
  generator->apply(weights);

  generator->to(device);

  SeqDiscriminator seq_discriminator(0.2);
  auto & discriminator = seq_discriminator.GetDiscriminator();
  discriminator->apply(weights);

  discriminator->to(device);

  auto dataset = ImageFolder(default_image_path, '_', image_size, image_size)
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
  std::cout << "batches_per_epoch: " << batches_per_epoch << std::endl;
  int64_t checkpoint_counter = 1;

  torch::Tensor initial_batch_image_real;
  bool got_first = false;

  for(size_t epoch=1; epoch<=knumber_of_epochs; ++epoch)
  {
    int64_t batch_index = 0;

    for(auto& batch : *data_loader)
    {
      // train discriminator

      // real images
      discriminator->zero_grad();
      torch::Tensor real_images = batch.data.to(device);

      if (!got_first)
      {
        initial_batch_image_real = real_images.clone();
        got_first = true;
      }

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
        std::cout << "[" << checkpoint_counter << "/" << (batches_per_epoch*knumber_of_epochs)  << "] " << "[" << epoch << "/" << knumber_of_epochs << "] [" << batch_index << "/" << batches_per_epoch << "] D_loss: " << d_loss.item<float>() << " G_loss: " << g_loss.item<float>() << " -- batch_sizes: " << batch.data.sizes() << "\n";
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
        dcgan_utils::RawImageData raw_fakeimage_output = dcgan_utils::ConvertTensorToRawImage(samples, 0, 0, false);

        const std::string output_path = "output_images/test" + std::to_string(checkpoint_counter);
        dcgan_utils::SaveRawImageDataToFile(output_path, raw_fakeimage_output, dcgan_utils::IMAGE_OUTPUT_TYPE::PNG);
#endif

        std::cout << "\n-> checkpoint " << checkpoint_counter << "\n\n";
      }

      if ((checkpoint_counter % kdisplay_interval) == 0)
      {
        torch::Tensor samples_fake = generator->forward(torch::randn({knumber_of_samples_per_checkpoint, klaten, 1, 1}, device));
        dcgan_utils::RawImageData raw_fakeimage_output = dcgan_utils::ConvertTensorToRawImage(samples_fake, 0, 0, false);

        torch::Tensor samples_real = initial_batch_image_real.clone();
        samples_real = samples_real.resize_({64,3,64,64});
        dcgan_utils::RawImageData raw_realimage_output = dcgan_utils::ConvertTensorToRawImage(samples_real, 0, 0, true);

        window.AddRawImageReals(raw_realimage_output);
        window.AddRawImageFakes(raw_fakeimage_output);

        window.AddDCGANPoint(checkpoint_counter, {d_loss.item<double>(), g_loss.item<double>(), real_output[real_output.size(0)-1].item<double>(), fake_output[fake_output.size(0)-1].item<double>()});
      }
    }
  }

  std::cout << "training complete!" << std::endl;

  std::this_thread::sleep_for(std::chrono::seconds(1));
  window.Close();

  return 0;
}