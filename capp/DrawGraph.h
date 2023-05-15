#pragma once

#include <cstdint>
#include <vector>
#include "DCGANUtils.h"

#include <sfml/Graphics.hpp>

class DrawGraph {
  public:
    DrawGraph() = delete;
    explicit DrawGraph(std::vector<dcgan_utils::DataPlot> & plot_data);

    void SetPosition(int32_t x, int32_t y);
    void SetShift(int32_t value);
    void SetMaxPlotValues(int32_t max);
    void Draw(sf::RenderWindow * window);

  private:
    int32_t x = 0;
    int32_t y = 0;
    std::vector<dcgan_utils::DataPlot> & plotData;
    int32_t maxPlotData = 100;
    int32_t shift = 0;
};
