#include "DCGANUtils.h"

#ifdef __clang__
#define STBIWDEF static inline
#endif
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <algorithm>
#include <fstream>

namespace {

void norm_ip(torch::Tensor & img, double low, double high)
{
  img = img.clamp(low, high);
  img = img.sub_(low).div_(std::max(high - low, 1e-5));
}

void norm_range(torch::Tensor & t)
{
  norm_ip(t, t.min().item<double>(), t.max().item<double>());
}

}

namespace dcgan_utils {

  RawImageData ConvertTensorToRawImage(const torch::Tensor & tensor, int32_t padding, int32_t pad_value)
  {
    std::cout << "input tensor w: " << tensor.size(3) << " h: " << tensor.size(2) << " c: " << tensor.size(1) << " layers:" << tensor.size(0) << " dim: " << tensor.dim() << std::endl;

    torch::Tensor cp_tensor = tensor.clone();
    norm_range(cp_tensor);

    int64_t nmaps = cp_tensor.size(0);
    int64_t xmaps = 8;
    auto ymaps = static_cast<int64_t>(std::ceil(static_cast<double>(nmaps) / static_cast<double>(xmaps)));
    int64_t height = cp_tensor.size(2) + padding;
    int64_t width = cp_tensor.size(3) + padding;
    int64_t nchannels = cp_tensor.size(1);

    int64_t k=0;
    auto grid = cp_tensor.new_full({nchannels, (height*ymaps + padding), (width*xmaps + padding)}, pad_value-1);
    for (int64_t y=0; y<ymaps; y++)
    {
        for (int64_t x=0; x<xmaps; x++)
        {
            if (k >= nmaps)
            {
                break;
            }

            grid.narrow(1, y*height+padding, height-padding).narrow(2, x*width+padding, width-padding).copy_(tensor[k]);
            k++;
        }
    }

    grid = (grid + 1) / 2.0;
    grid = grid.squeeze().detach();
    grid = grid.permute({1,2,0}).contiguous();
    grid = grid.mul(255).clamp(0, 255).to(torch::kU8);
    grid = grid.to(torch::kCPU);

    std::cout << "output grid tensor w: " << grid.size(0) << " h: " << grid.size(1) << " c: " << grid.size(2) << " size: " << grid.sizes() << std::endl;

    auto output_data = grid.data_ptr<uint8_t>();
    int64_t output_width = grid.size(0);
    int64_t output_height = grid.size(1);
    int64_t output_color_depth = grid.size(2);
    int64_t byte_size = output_height * output_width * output_color_depth;

    RawImageData raw_image_data;
    raw_image_data.tensor = grid;
    raw_image_data.data = output_data;
    raw_image_data.width = output_width;
    raw_image_data.height = output_height;
    raw_image_data.color_depth = output_color_depth;
    raw_image_data.byte_size = byte_size;

    return raw_image_data;
  }

  void SaveRawImageDataToFile(std::string file_path, const RawImageData & raw_image_data, IMAGE_OUTPUT_TYPE output_type)
  {
    auto image_width = static_cast<int32_t>(raw_image_data.width);
    auto image_height = static_cast<int32_t>(raw_image_data.height);
    auto image_color_depth = static_cast<int32_t>(raw_image_data.color_depth);

    switch(output_type)
    {
        case IMAGE_OUTPUT_TYPE::JPG:
          stbi_write_jpg((file_path + ".jpg").c_str(), image_width, image_height, image_color_depth, raw_image_data.data, 100);
        break;

        case IMAGE_OUTPUT_TYPE::PNG:
          stbi_write_png((file_path + ".png").c_str(), image_width, image_height, image_color_depth, raw_image_data.data, (image_width*image_color_depth));
        break;

        case IMAGE_OUTPUT_TYPE::RAW:
        default:
        {
          std::ofstream f_out(file_path + ".raw");
          f_out.write(reinterpret_cast<char*>(raw_image_data.data), raw_image_data.byte_size);
          f_out.flush();
          f_out.close();
        }
    }
  }
}