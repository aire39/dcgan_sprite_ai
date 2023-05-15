#include "DrawGraph.h"

DrawGraph::DrawGraph(std::vector<dcgan_utils::DataPlot> & plot_data)
  : plotData(plot_data)
{
}

void DrawGraph::SetPosition(int32_t x, int32_t y)
{
  this->x = x;
  this->y = y;
}

void DrawGraph::SetShift(int32_t value)
{
  shift = value;
}

void DrawGraph::SetMaxPlotValues(int32_t max)
{
  maxPlotData = max;
}

void DrawGraph::Draw(sf::RenderWindow * window)
{
  // render
  sf::CircleShape point(4.0f);

  sf::VertexArray vertices_d_loss;
  vertices_d_loss.setPrimitiveType(sf::PrimitiveType::LinesStrip);

  sf::VertexArray vertices_g_loss;
  vertices_g_loss.setPrimitiveType(sf::PrimitiveType::LinesStrip);

  sf::VertexArray vertices_d_prob;
  vertices_d_prob.setPrimitiveType(sf::PrimitiveType::LinesStrip);

  sf::VertexArray vertices_g_conf;
  vertices_g_conf.setPrimitiveType(sf::PrimitiveType::LinesStrip);

  for (const auto & data : plotData)
  {
    /////

    point.setFillColor(sf::Color::Red);
    point.setPosition((x + data.x) - shift - 4.0f, (y + -(data.dcganValues.d_prob * 20)));
    vertices_d_prob.append(sf::Vertex(sf::Vector2f((x + data.x) - shift, (y + -(data.dcganValues.d_prob * 20.0f))), sf::Color::Red));

    window->draw(&vertices_d_loss[0], vertices_d_loss.getVertexCount(), vertices_d_loss.getPrimitiveType());
    window->draw(point);

    point.setFillColor(sf::Color::Magenta);
    point.setPosition((x + data.x) - shift - 4.0f, (y + -data.dcganValues.dg_conf * 20));
    vertices_g_conf.append(sf::Vertex(sf::Vector2f((x + data.x) - shift, (y + -(data.dcganValues.dg_conf * 20.0f))), sf::Color::Magenta));

    window->draw(&vertices_g_loss[0], vertices_g_loss.getVertexCount(), vertices_g_loss.getPrimitiveType());
    window->draw(point);

    /////

    point.setFillColor(sf::Color::Green);
    point.setPosition((x + data.x) - shift - 4.0f, (y + -(data.dcganValues.d_loss * 20)));
    vertices_d_loss.append(sf::Vertex(sf::Vector2f((x + data.x) - shift, (y + -(data.dcganValues.d_loss * 20.0f))), sf::Color::Green));

    window->draw(&vertices_d_loss[0], vertices_d_loss.getVertexCount(), vertices_d_loss.getPrimitiveType());
    window->draw(point);

    point.setFillColor(sf::Color::Blue);
    point.setPosition((x + data.x) - shift - 4.0f, (y + -data.dcganValues.g_loss * 20));
    vertices_g_loss.append(sf::Vertex(sf::Vector2f((x + data.x) - shift, (y + -(data.dcganValues.g_loss * 20.0f))), sf::Color::Blue));

    window->draw(&vertices_g_loss[0], vertices_g_loss.getVertexCount(), vertices_g_loss.getPrimitiveType());
    window->draw(point);
  }
}