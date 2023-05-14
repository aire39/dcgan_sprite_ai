#include "DCGANUtils.h"

#include <fstream>

namespace dcgan_utils {

  RawImageData ConvertTensorToRawImage(const torch::Tensor & tensor, int32_t padding, int32_t pad_value)
  {
    std::cout << "input tensor w: " << tensor.size(3) << " h: " << tensor.size(2) << " c: " << tensor.size(1) << " layers:" << tensor.size(0) << " dim: " << tensor.dim() << std::endl;

    int64_t nmaps = tensor.size(0);
    int64_t xmaps = 8;
    auto ymaps = static_cast<int64_t>(std::ceil(static_cast<double>(nmaps) / static_cast<double>(xmaps)));
    int64_t height = tensor.size(2) + padding;
    int64_t width = tensor.size(3) + padding;
    int64_t nchannels = tensor.size(1);

    int64_t k=0;
    auto grid = tensor.new_full({nchannels, (height*ymaps + padding), (width*xmaps + padding)}, pad_value-1);
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

    std::cout << "output grid tensor w: " << grid.size(2) << " h: " << grid.size(1) << " c: " << grid.size(0) << " size: " << grid.sizes() << std::endl;

    auto output_data = grid.data_ptr<uint8_t>();
    int64_t output_width = grid.size(2);
    int64_t output_height = grid.size(1);
    int64_t output_color_depth = grid.size(0);

    int64_t byte_size = output_height * output_width * output_color_depth;

    return {grid, output_data, output_width, output_height, output_color_depth, byte_size};
  }

  void SaveRawImageDataToFile(std::string file_path, const RawImageData & raw_image_data)
  {
    std::ofstream f_out(file_path);
    f_out.write(reinterpret_cast<char*>(raw_image_data.data), raw_image_data.byte_size);
    f_out.flush();
    f_out.close();
  }
}