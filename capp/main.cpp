#include <iostream>
#include <torch/torch.h>
#include <filesystem>
#include "GeneratorImpl.h"
#include "SeqDiscriminator.h"
#include "ImageFolder.h"
#include "DCGANUtils.h"
#include "Window.h"
#include "CLI11/include/CLI/CLI.hpp"

#define SAVE_IMAGES_RGB true
#if SAVE_IMAGES_RGB
#include <fstream>
#endif


int main([[maybe_unused]] int argc, [[maybe_unused]] char*argv[])
{
  CLI::App app ("DCGAN Application");

  std::string source_directory = "data/creatures/images";
  app.add_option("-s,--source_dir", source_directory, "location of source directory that holds images. default dir: data/creatures/images");

  std::string output_directory = "output_images";
  app.add_option("-o,--output_dir", output_directory, "location of output directory where results need to be saved. default dir: output_images");

  if(!std::filesystem::is_directory(output_directory))
  {
    std::filesystem::create_directories(output_directory);
  }

  std::string file_output_name = "test";
  app.add_option("-f,--output_file", file_output_name, "location of output filename (no need to add extension)");

  int32_t file_type = 2;
  app.add_option("-t,--out_file_type", file_type, "Enter number value --> (0) RAW, (1) JPG, (2) PNG");
  file_type = std::clamp(file_type, 0, 2);

  bool show_gui = false;
  app.add_flag("-g,--gui", show_gui, "show dcgan results running in gui window");

  int64_t gui_update = 10;
  app.add_option("-u,--gui_update", gui_update, "update frequency of gui window. how many samples must be completed before viewing");

  int64_t user_log_interval = 4;
  app.add_option("-i,--info_int", user_log_interval, "interval print info. how many itereations before we see a print of data from the training. default: 4");

  bool use_weights = false;
  app.add_flag("-w,--use_weights", use_weights, "add weights values");

  double learn_rate = 0.0002;
  app.add_flag("-l,--learn_rate", learn_rate, "set learn_rate for network. default: 0.0002");

  CLI11_PARSE(app, argc, argv)

  // Setup Window To See DCGAN running in real time

  Window window;

  if (show_gui)
  {
    window.Run();
  }

  // DCGAN Setup and Training

  constexpr int32_t image_size = 64; // WxH
  constexpr int64_t knoise_size = 100;
  constexpr int64_t kbatch_size = 128;
  constexpr int32_t knumber_of_workers = 2;
  constexpr int32_t knumber_of_epochs = 1000;
  constexpr bool kenforce_order = false;
  const double klr = learn_rate;
  constexpr double kbeta1 = 0.5;
  constexpr double kbeta2 = 0.999;
  constexpr int64_t klaten = 100;
  const int64_t klog_interval = user_log_interval;
  constexpr int64_t kcheckpoint_interval = 200;
  int64_t kdisplay_interval = gui_update;
  constexpr int64_t knumber_of_samples_per_checkpoint = 64;
  constexpr double knegative_slope = 0.2;
  const char* default_image_path = source_directory.c_str();
  const char* default_output_path = output_directory.c_str();

  torch::manual_seed(999);

  torch::Device device(torch::kCPU);

  if (torch::cuda::is_available())
  {
    device = torch::Device(torch::kCUDA);
    std::cout << "Cuda available! Will use found GPU!" << std::endl;
  }

  auto weights = torch::nn::Module::NamedModuleApplyFunction([](const std::string & name, torch::nn::Module & m){
      std::cout << "weight name: " << m.name() << std::endl;

      if (m.name().find("Conv") != std::string::npos)
      {
        for (auto & p : m.named_parameters())
        {
          if (p.key() == "weight")
          {
            torch::nn::init::normal_(p->data(), 0.0, 0.02);
          }
        }
      }
      else if (m.name().find("BatchNorm") != std::string::npos)
      {
        for (auto & p : m.named_parameters())
        {
          if (p.key() == "weight")
          {
            torch::nn::init::normal_(p->data(), 1.0, 0.02);
          }
          else if (p.key() == "bias")
          {
            torch::nn::init::constant_(p->data(), 0.0);
          }
        }
      }
  });

  Generator generator (knoise_size);
  generator->to(device);

  if (use_weights)
  {
    generator->apply(weights);
  }

  SeqDiscriminator seq_discriminator(knegative_slope);
  auto & discriminator = seq_discriminator.GetDiscriminator();
  discriminator->to(device);

  if(use_weights)
  {
    discriminator->apply(weights);
  }

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
      torch::Tensor real_output = discriminator->forward(real_images).reshape({real_labels.sizes()});
      torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
      d_loss_real.backward();
      auto d_x = real_output.mean().item<double>();

      // fake images
      torch::Tensor noise = torch::randn({batch.data.size(0), klaten, 1, 1}, device);
      torch::Tensor fake_images = generator->forward(noise);
      torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);

      torch::Tensor fake_output = discriminator->forward(fake_images.detach()).reshape({fake_labels.sizes()});
      torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
      d_loss_fake.backward();

      auto dg_x = fake_images.mean().item<double>();

      torch::Tensor d_loss = d_loss_real + d_loss_fake;
      discriminator_optimizer.step();

      // train generator

      generator->zero_grad();
      fake_labels = fake_labels.fill_(1);
      fake_output = discriminator->forward(fake_images).reshape({fake_labels.sizes()});
      torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
      g_loss.backward();

      auto dg_x2 = fake_output.mean().item<double>();

      generator_optimizer.step();

      // end of training step

      batch_index++;
      checkpoint_counter++;

      if ((batch_index % klog_interval) == 0)
      {
        std::cout << "[" << checkpoint_counter << "/" << (batches_per_epoch*knumber_of_epochs)  << "] " << "[" << epoch << "/" << knumber_of_epochs << "] [" << batch_index << "/" << batches_per_epoch << "] D_loss: " << d_loss.item<float>() << " G_loss: " << g_loss.item<float>() << " D_x: " << d_x << " DG_x: " << dg_x << " / " << dg_x2 << " -- batch_sizes: " << batch.data.sizes() << "\n";
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

        const std::string output_path = std::string(default_output_path) + "/" + file_output_name + std::to_string(checkpoint_counter);


        dcgan_utils::SaveRawImageDataToFile(output_path, raw_fakeimage_output, (dcgan_utils::IMAGE_OUTPUT_TYPE)file_type);
#endif

        std::cout << "\n-> checkpoint " << checkpoint_counter << "\n\n";
      }

      if (((checkpoint_counter % kdisplay_interval) == 0) && show_gui)
      {
        torch::Tensor samples_fake;
        {
          torch::NoGradGuard no_grad;
          samples_fake = generator->forward(torch::randn({knumber_of_samples_per_checkpoint, klaten, 1, 1}, device));
        }
        dcgan_utils::RawImageData raw_fakeimage_output = dcgan_utils::ConvertTensorToRawImage(samples_fake, 0, 0, false);

        torch::Tensor samples_real = initial_batch_image_real.clone();
        samples_real = samples_real.resize_({64,3,64,64});
        dcgan_utils::RawImageData raw_realimage_output = dcgan_utils::ConvertTensorToRawImage(samples_real, 0, 0, true);

        window.AddRawImageReals(raw_realimage_output);
        window.AddRawImageFakes(raw_fakeimage_output);

        window.AddDCGANPoint(static_cast<int32_t>(checkpoint_counter), {d_loss.item<double>(), g_loss.item<double>(), d_x, dg_x2});
      }
    }
  }

  std::cout << "training complete!" << std::endl;

  std::this_thread::sleep_for(std::chrono::seconds(1));
  window.Close();

  return 0;
}