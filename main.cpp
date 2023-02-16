#include "include.hpp"
#include "Renderer.cpp"


int main() noexcept{
  spdlog::set_level(spdlog::level::trace);

  if(not glfwInit() and not glfwVulkanSupported()) std::abort();

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  auto const window = glfwCreateWindow(690, 690, "wing boo のまちとりしのはまりちのとしま", nullptr, nullptr);
  if(not window) std::abort();

  auto renderer = create_renderer(window); 
  spdlog::info("Created renderer {}",1);

  for(;not glfwWindowShouldClose(window);){
    glfwPollEvents();
    using namespace std::chrono_literals;
    renderer.draw_frame();
    std::this_thread::sleep_for(16ms);
  }
  glfwTerminate();
}


