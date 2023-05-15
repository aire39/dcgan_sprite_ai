#pragma once

#include <string>
#include <cstdint>
#include <torch/torch.h>

namespace dcgan_utils {
  enum class IMAGE_OUTPUT_TYPE {RAW, JPG, PNG};

  struct RawImageData
  {
    torch::Tensor tensor;
    uint8_t * data = nullptr;
    int64_t width = 0;
    int64_t height = 0;
    int64_t color_depth = 0;
    int64_t byte_size = 0;
  };

  RawImageData ConvertTensorToRawImage(const torch::Tensor & tensor, int32_t padding, int32_t pad_value);
  void SaveRawImageDataToFile(std::string file_path, const RawImageData & raw_image_data, IMAGE_OUTPUT_TYPE output_type);
}