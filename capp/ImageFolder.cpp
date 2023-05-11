//
// Created by Sean on 5/11/2023.
//

#include "ImageFolder.h"

#include <iostream>
#include <filesystem>
#include "cimg/CImg.h"

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

ImageFolder::ImageFolder(std::string folder_path, char deliminator)
{
  images = loadfiles(folder_path, deliminator);
}

torch::data::Example<> ImageFolder::get(size_t index)
{
  auto [image_path, image_ext, image_label] = images[index];

  cimg_library::CImg<uint8_t> image(image_path.c_str());
  image.resize(64, 64, 1, 3);

  torch::Tensor img_tensor = torch::from_blob(image.data(), {image.width(), image.height(), 3}, torch::kByte).clone();
  img_tensor = img_tensor.permute({2, 0, 1});

  torch::Tensor label_tensor = torch::full({1}, image_label);

  return {img_tensor, label_tensor};
}

torch::optional<size_t> ImageFolder::size() const
{
  return images.size();
}