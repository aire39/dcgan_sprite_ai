#pragma once

#include <memory>
#include <thread>
#include <queue>
#include <sfml/Graphics.hpp>
#include <mutex>
#include "DCGANUtils.h"
#include "DrawGraph.h"

class Window {
  public:
    Window() = default;
    ~Window();

    void Run();
    void Close();
    void AddRawImageFakes(const dcgan_utils::RawImageData& raw_image);
    void AddRawImageReals(dcgan_utils::RawImageData raw_image);
    void AddDCGANPoint(int32_t x, dcgan_utils::DataPlot::dcgan_values values);

  private:
    std::mutex addImageMtx;
    std::mutex addPointMtx;
    dcgan_utils::RawImageData realImage;
    std::queue<dcgan_utils::RawImageData> qFakeImages;
    std::vector<dcgan_utils::DataPlot> graphData;
    std::unique_ptr<sf::RenderWindow> window;
    std::thread windowThread;
    bool isRunning = true;
    std::unique_ptr<DrawGraph> graph;

    void WindowTask();
};
