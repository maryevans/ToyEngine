#include "include.hpp"

class Renderer{
public:
  [[nodiscard]]
  Renderer(GLFWwindow * window) noexcept;
  ~Renderer();

  vk::UniqueInstance instance;
#ifdef DEBUG
  vk::DebugUtilsMessengerUnique messenger;
#endif

  vk::UniqueDevice device;

  Renderer(Renderer const &) = delete;
  Renderer(Renderer &&) = delete;
  Renderer & operator=(Renderer const &) = delete;
  Renderer & operator=(Renderer &&) = delete;
};

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    std::cout << "validation layer: " << pCallbackData->pMessage << "\n\n";

    return VK_FALSE;
}

Renderer::Renderer(GLFWwindow * window) noexcept{

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

  instance = vk::createInstanceUnique(instanceInfo);


#ifdef DEBUG
  auto const messengerInfo = vk::DebugUtilsMessengerCreateInfoEXT{
    .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral,
    .pfnUserCallback = debugCallback,
    .pUserData = nullptr
  };

  auto messenger = instance->createDebugUtilsMessengerEXTUnique(messengerInfo);
#endif

  auto const surface = std::invoke([&]{
      VkSurfaceKHR surface;
      if(glfwCreateWindowSurface(*instance, window, nullptr, &surface)){
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

      for(auto physical_device : instance->enumeratePhysicalDevices())
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
        if(indices.graphicsIndex < 0 and properties[i].queueFlags & vk::QueueFlagBits::eGraphics) indices.graphicsIndex = i;
          
        if(indices.presentIndex < 0 and physical_device.getSurfaceSupportKHR(1, surface)) indices.presentIndex = i;

        if(indices.graphicsIndex >= 0 and indices.presentIndex >= 0){
          return indices;
        }
      }

      spdlog::error("Unable to find graphics card that can display");
      std::abort();
  });

  static auto graphicsQueuePriority = 1.0f;

  auto const queueCreateInfos = {
    vk::DeviceQueueCreateInfo{
      .queueFamilyIndex = (uint32_t)graphicsFamilyIndex,
      .queueCount = 1,
      .pQueuePriorities = &graphicsQueuePriority,
    },
  };

  auto const 

  device = physical_device.createDeviceUnique(
      vk::DeviceCreateInfo{
        .enabledLayerCount
      }

  );
}

Renderer::~Renderer(){ }

