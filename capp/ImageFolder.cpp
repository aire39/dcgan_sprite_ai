//
// Created by Sean on 5/11/2023.
//

#include "ImageFolder.h"

#include <iostream>
#include <filesystem>

#define SAVE_RAW_INPUT_IMAGE_DATA false
#define PRINT_IMAGE_COUNTER false

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_resize.h"

#if SAVE_RAW_INPUT_IMAGE_DATA
#include <fstream>
#endif

namespace {
  std::vector<std::tuple<std::string,std::string,int64_t>> loadfiles(std::string & folder_path, char deliminator)
  {
    std::vector<std::tuple<std::string,std::string,int64_t>> files;

    for(auto const& entry : std::filesystem::directory_iterator(folder_path))
    {
      std::string image_path {entry.path().generic_u8string()};

      // grab extension
      size_t find_start_of_extension = image_path.find_last_of('.') + 1;
      std::string extension = image_path.substr(find_start_of_extension);

      // grab the image index value from the name which is expected happen after the last deliminator character
      size_t last_deliminator_char = image_path.find_last_of(deliminator) + 1;
      std::string image_index = image_path.substr(last_deliminator_char);

      size_t last_dot_char_pos = image_index.find_first_of('.');
      image_index = image_index.substr(0, last_dot_char_pos);

      int64_t label = 0;

      try {
          label = std::stoll(image_index);
      }
      catch (const std::invalid_argument & ex) {
          std::cerr << "Unable to determine label value for " << image_path << " --> exception raised: " << ex.what() << std::endl;
          break;
      }

      files.emplace_back(image_path, extension, label);
    }

    return files;
  }
}

ImageFolder::ImageFolder(std::string folder_path, char deliminator, int32_t expected_width, int32_t expected_height)
  : expectedWidth (expected_width)
  , expectedHeight (expected_height)
{
  images = loadfiles(folder_path, deliminator);
}

torch::data::Example<> ImageFolder::get(size_t index)
{
  numberUsed++;

#if PRINT_IMAGE_COUNTER
  std::cout << numberUsed << "\n";
#endif

  auto [image_path, image_ext, image_label] = images[index];

  constexpr int64_t desired_num_channels = 3;

  int32_t width, height, channels;
  uint8_t *image = stbi_load(image_path.data(), &width, &height, &channels, desired_num_channels);

  int32_t new_width = expectedWidth;
  int32_t new_height = expectedHeight;
  int32_t new_channel = desired_num_channels;

  std::vector<uint8_t> uimage(new_width*new_height*new_channel);

  stbir_resize(image
              ,width
              ,height
              ,0
              ,uimage.data()
              ,new_width
              ,new_height
              ,0
              ,STBIR_TYPE_UINT8
              ,new_channel
              ,-1
              ,0
              ,STBIR_EDGE_CLAMP
              ,STBIR_EDGE_CLAMP
              ,STBIR_FILTER_DEFAULT
              ,STBIR_FILTER_DEFAULT
              ,STBIR_COLORSPACE_LINEAR
              ,nullptr);

  torch::Tensor img_tensor = torch::from_blob(uimage.data(), {new_height, new_width, new_channel}, torch::kByte).clone();
  img_tensor = img_tensor.permute({2, 0, 1});

  torch::Tensor label_tensor = torch::full({1}, image_label);

#if SAVE_RAW_INPUT_IMAGE_DATA
  std::cout << "loading image: " << image_path << " width: " << new_width << " height: " << new_height << " size: " << uimage.size() << "\n";
  std::ofstream output("test/test_" + std::to_string(image_label) + ".rgb");
  output.write(reinterpret_cast<char*>(uimage.data()), (new_width*new_height*new_channel));
  output.close();
#endif

  return {img_tensor, label_tensor};
}

torch::optional<size_t> ImageFolder::size() const
{
  return images.size();
}