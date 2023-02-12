#pragma once
#include "include.hpp"

class Renderer{
public:

  void draw_frame();
  friend inline auto create_renderer(GLFWwindow * window) noexcept;

  Renderer(Renderer && renderer):
    instance(std::move(renderer.instance)),
#ifdef DEBUG
    messenger(std::move(renderer.messenger)),
#endif
    device(std::move(renderer.device))
  { }

  Renderer & operator=(Renderer && renderer){
    instance = std::move(renderer.instance);
#ifdef DEBUG
    messenger = std::move(renderer.messenger);
#endif
    device = std::move(renderer.device);

    return *this;
  }

  ~Renderer(){}
private:
  [[nodiscard]] Renderer() noexcept{}

  vk::UniqueInstance instance;
#ifdef DEBUG
  vk::DebugUtilsMessengerUnique messenger;
#endif
  vk::UniqueDevice device;

  Renderer(Renderer const &) = delete;
  Renderer & operator=(Renderer const &) = delete;
};

#ifdef DEBUG
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    if(messageSeverity & VkDebugUtilsMessageSeverityFlagBitsEXT::VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) spdlog::error(pCallbackData->pMessage);
    if(messageSeverity & VkDebugUtilsMessageSeverityFlagBitsEXT::VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) spdlog::warn(pCallbackData->pMessage);

    std::cout << "validation layer: " << pCallbackData->pMessage << "\n\n";

    return VK_FALSE;
}
#endif

[[nodiscard]]
inline auto create_renderer(GLFWwindow * window) noexcept{
  auto renderer = Renderer();
  constexpr auto appinfo = vk::ApplicationInfo{
    .pApplicationName = "Toy",
    .applicationVersion = 0,
    .pEngineName = "Toy",
    .engineVersion = 0,
    .apiVersion = VK_API_VERSION_1_3,
  };

  //TODO: don't allocate here.
  auto const extensions = std::invoke([]{
    auto glfwExtensionCount = uint32_t(0);
    auto const glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    auto extensions = std::vector<const char *>(glfwExtensions, glfwExtensions + glfwExtensionCount);

#ifdef DEBUG
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

    return extensions;
  });


  auto const layers = "VK_LAYER_KHRONOS_validation";

  auto const instanceInfo = vk::InstanceCreateInfo{
    .pApplicationInfo = &appinfo,
    .enabledLayerCount = 1,
    .ppEnabledLayerNames = &layers,
    .enabledExtensionCount = (uint32_t) extensions.size(),
    .ppEnabledExtensionNames = extensions.data(),
  };

  renderer.instance = vk::createInstanceUnique(instanceInfo);


#ifdef DEBUG
  auto const messengerInfo = vk::DebugUtilsMessengerCreateInfoEXT{
    .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral,
    .pfnUserCallback = debugCallback,
    .pUserData = nullptr
  };

  auto messenger = renderer.instance->createDebugUtilsMessengerEXTUnique(messengerInfo);
#endif

  auto const surface = std::invoke([&]{
      VkSurfaceKHR surface;
      if(glfwCreateWindowSurface(*renderer.instance, window, nullptr, &surface)){
        spdlog::error("Unable to create window surface");
        std::abort();
      }

      return vk::SurfaceKHR(surface);
  });

  auto const physical_device = std::invoke([&]{
      //TODO:
      auto constexpr physical_device_has_ideal_properties = [](vk::PhysicalDevice const & device){
        return true;
      };

      for(auto physical_device : renderer.instance->enumeratePhysicalDevices())
      {
        if(physical_device_has_ideal_properties(physical_device)){
          return physical_device;
        }
      }
      spdlog::error("No viable gpu");
      std::abort();
  });

  auto const [ graphicsFamilyIndex, displayFamilyIndex ] = std::invoke([&] noexcept{ 
      struct {
        int graphicsIndex = -1;
        int presentIndex = -1;
      } indices; 

      auto const properties = physical_device.getQueueFamilyProperties();

      for(auto i = 0; i < properties.size(); ++i){
        if(indices.graphicsIndex < 0 and properties[i].queueFlags & vk::QueueFlagBits::eGraphics){
          spdlog::info("Found graphics index {}", 1);
          indices.graphicsIndex = i;
        }
          
        if(indices.presentIndex < 0 and physical_device.getSurfaceSupportKHR(1, surface)) {
          //spdlog::info("Found present iindex {}", i);
          indices.presentIndex = i;
        }

        if(indices.graphicsIndex >= 0 and indices.presentIndex >= 0){
          return indices;
        }
      }

      spdlog::error("Unable to find graphics card that can display");
      std::abort();
  });

  static auto graphicsQueuePriority = 1.0f;

  auto const queueCreateInfos = std::array{
    vk::DeviceQueueCreateInfo{
      .queueFamilyIndex = (uint32_t)graphicsFamilyIndex,
      .queueCount = 1,
      .pQueuePriorities = &graphicsQueuePriority,
    },
  };

  auto const device_extensions = std::array{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  renderer.device = physical_device.createDeviceUnique(
      vk::DeviceCreateInfo{ 
        .queueCreateInfoCount = queueCreateInfos.size(),
        .pQueueCreateInfos = queueCreateInfos.data(),
        .enabledLayerCount = 1,
        .ppEnabledLayerNames = &layers,
        .enabledExtensionCount = device_extensions.size(),
        .ppEnabledExtensionNames = device_extensions.data()
      });

  auto const graphicsQueue = renderer.device->getQueue(graphicsFamilyIndex, 0);




  return std::move(renderer);
}



void Renderer::draw_frame(){

}
