#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <torch/torch.h>

class ImageFolder : public torch::data::Dataset<ImageFolder> {
  public:
    ImageFolder() = delete;
    explicit ImageFolder(std::string folder_path, char deliminator);

    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;

  private:
    std::vector<std::tuple<std::string,std::string,int64_t>> images; // tuple: image file path, label

};
