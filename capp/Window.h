#pragma once

#include <memory>
#include <thread>
#include <queue>
#include <sfml/Graphics.hpp>
#include <mutex>
#include "DCGANUtils.h"

class Window {
  public:
    Window() = default;
    ~Window();

    void Run();
    void AddRawImageFakes(const dcgan_utils::RawImageData& raw_image);
    void AddRawImageReals(dcgan_utils::RawImageData raw_image);

  private:
    std::mutex addImageMtx;
    dcgan_utils::RawImageData realImage;
    std::queue<dcgan_utils::RawImageData> qFakeImages;
    std::unique_ptr<sf::RenderWindow> window;
    std::thread windowThread;

    void WindowTask();
};
