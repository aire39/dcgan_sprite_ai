//
// Created by Sean on 5/14/2023.
//

#include "Window.h"

#include <chrono>
#include <sfml/OpenGL.hpp>
#include <utility>

Window::~Window()
{
  if(windowThread.joinable())
  {
    windowThread.join();
  }
}

void Window::Run()
{
  windowThread = std::thread(&Window::WindowTask, this);
}

void Window::AddRawImageFakes(const dcgan_utils::RawImageData& raw_image)
{
  std::lock_guard<std::mutex> lock(addImageMtx);
  qFakeImages.push(raw_image);
}

void Window::AddRawImageReals(dcgan_utils::RawImageData raw_image)
{
    std::lock_guard<std::mutex> lock(addImageMtx);
    realImage = std::move(raw_image);
}

void Window::WindowTask()
{
  window = std::make_unique<sf::RenderWindow>(sf::VideoMode(1600, 600), "DCGAN Results");
  window->setActive(true);

  sf::Texture texture_fake;
  texture_fake.create(512, 512);

  sf::Image img_fake;
  img_fake.create(512, 512, sf::Color::White);

  texture_fake.update(img_fake);

  sf::Texture texture_real;
  texture_real.create(512, 512);

  sf::Image img_real;
  img_real.create(512, 512, sf::Color::White);

  texture_real.update(img_real);
  
  sf::Sprite sprite_fake;
  sprite_fake.setTexture(texture_fake);
  sprite_fake.setPosition((144 + 800 ), 44);

  sf::Sprite sprite_real;
  sprite_real.setTexture(texture_real);
  sprite_real.setPosition(144, 44);

  bool is_running = true;

  while(is_running)
  {
    sf::Event event{};

    while(window->pollEvent(event))
    {
      if (event.type == sf::Event::Closed)
      {
        window->close();
        is_running = false;
      }
      else if (event.type == sf::Event::Resized)
      {
        glViewport(0, 0, static_cast<int32_t>(event.size.width), static_cast<int32_t>(event.size.height));
      }
    }

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    {
      std::lock_guard<std::mutex> lock(addImageMtx);

      if (!qFakeImages.empty())
      {
        auto raw_image = qFakeImages.front();

        if (!raw_image.data)
        {
          qFakeImages.pop();
          continue;
        }

        for(int i=0; i<raw_image.height; i++)
          for(int j=0; j<raw_image.width; j++)
          {
            int red = ((j*(int)raw_image.color_depth)+0) + i*(int)(raw_image.width*raw_image.color_depth);
            int green = ((j*(int)raw_image.color_depth)+1) + i*(int)(raw_image.width*raw_image.color_depth);
            int blue = ((j*(int)raw_image.color_depth)+2) + i*(int)(raw_image.width*raw_image.color_depth);

            img_fake.setPixel(j, i, sf::Color(raw_image.data[red], raw_image.data[green], raw_image.data[blue], 255));
            img_real.setPixel(j, i, sf::Color(realImage.data[red], realImage.data[green], realImage.data[blue], 255));
          }

        texture_fake.update(img_fake);
        texture_real.update(img_real);

        qFakeImages.pop();
      }
    }

    window->draw(sprite_fake);
    window->draw(sprite_real);

    window->display();

    std::this_thread::sleep_for(std::chrono::milliseconds(16));
  }
}