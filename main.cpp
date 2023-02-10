#include "include.hpp"

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    std::cout << "validation layer: " << pCallbackData->pMessage << "\n\n";

    return VK_FALSE;
}

int main() noexcept{
  if(not glfwInit() and not glfwVulkanSupported()) std::abort();

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  auto const window = glfwCreateWindow(690, 690, "wing boo のまちとりしのはまりちのとしま", nullptr, nullptr);
  if(not window) std::abort();

  constexpr auto appinfo = vk::ApplicationInfo{
    .pApplicationName = "Toy",
    .applicationVersion = 0,
    .pEngineName = "Toy",
    .engineVersion = 0,
    .apiVersion = VK_API_VERSION_1_3,
  };

  auto const extensions = std::invoke([]{
    auto glfwExtensionCount = uint32_t(0);
    auto const glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    auto extensions = std::vector<const char *>(glfwExtensions, glfwExtensions + glfwExtensionCount);

//#ifdef DEBUG
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
//#endif

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

  auto instance = vk::createInstanceUnique(instanceInfo);

//#ifdef DEBUG
  auto const messengerInfo = vk::DebugUtilsMessengerCreateInfoEXT{
    .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral,
    .pfnUserCallback = debugCallback,
    .pUserData = nullptr
  };

  auto messenger = instance->createDebugUtilsMessengerEXTUnique(messengerInfo);
//#endif

  auto const surface = std::invoke([&]{
      VkSurfaceKHR surface;
      if(glfwCreateWindowSurface(*instance, window, nullptr, &surface)){
        spdlog::error("Unable to create window surface");
        std::abort();
      }

      return vk::SurfaceKHR(surface);
  });

  //TODO:
  auto const physical_device = instance->enumeratePhysicalDevices().back();

  //TODO:
  auto const [
    graphicsFamilyIndex,
    displayFamilyIndex
  ] = std::invoke([&]{
      struct{ int bla = 1; int bla2 = 2;} bla;

      return bla;
  });




  for(;not glfwWindowShouldClose(window);){
    glfwPollEvents();
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(2ms);
  }

  std::cout << "bla" << std::endl;
  glfwTerminate();
}


