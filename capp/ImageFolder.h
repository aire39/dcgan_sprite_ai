#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <torch/torch.h>

class ImageFolder : public torch::data::Dataset<ImageFolder> {
  public:
    ImageFolder() = delete;
    explicit ImageFolder(std::string folder_path, char deliminator, int32_t expected_width, int32_t expected_height);

    torch::data::Example<> get(size_t index) override;
    [[nodiscard]] torch::optional<size_t> size() const override;

  private:
    std::vector<std::tuple<std::string,std::string,int64_t>> images; // tuple: image file path, label
    int64_t numberUsed = 0;
    int32_t expectedWidth = 0;
    int32_t expectedHeight = 0;
};
