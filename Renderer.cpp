#pragma once
#include "include.hpp"

struct Depth_Image_Handles {
  vk::UniqueImage image;
  vk::UniqueDeviceMemory memory;
  vk::UniqueImageView view;
};

struct Image_Handles{
  vk::UniqueImage image;
  vk::UniqueDeviceMemory memory;
};

class Renderer{
public:
  void draw_frame();
  friend inline auto create_renderer(GLFWwindow * window) noexcept;

  Renderer(Renderer && renderer):
    instance(std::move(renderer.instance)),
#ifdef DEBUG
    messenger(std::move(renderer.messenger)),
#endif
    device(std::move(renderer.device)),
    surface(std::move(renderer.surface)),
    swapchain(std::move(renderer.swapchain)),
    swapchain_image_views(std::move(renderer.swapchain_image_views)),
    render_pass(std::move(renderer.render_pass)),
    descriptor_set_layout(std::move(renderer.descriptor_set_layout)),
    pieplien_layout(std::move(renderer.pieplien_layout)),
    graphics_pipeline(std::move(renderer.graphics_pipeline)),
    depth_image_handles(std::move(renderer.depth_image_handles)),
    depth_buffers(std::move(renderer.depth_buffers)),
    command_pool(std::move(renderer.command_pool)),
    descriptor_pool(std::move(renderer.descriptor_pool)),
    descriptor_sets(std::move(renderer.descriptor_sets)),
    command_buffers(std::move(renderer.command_buffers)),
    vertex_buffer(std::move(renderer.vertex_buffer)),
    vertex_buffer_memory(std::move(renderer.vertex_buffer_memory)),
    index_buffer(std::move(renderer.index_buffer)),
    index_buffer_memory(std::move(renderer.index_buffer_memory)),
    uniform_buffers(std::move(renderer.uniform_buffers)),
    image_handles(std::move(renderer.image_handles)),
    texture_mipmap_levels(std::move(renderer.texture_mipmap_levels)),
    texture_image_view(std::move(renderer.texture_image_view)),
    texture_sampler(std::move(renderer.texture_sampler)),
    swapchain_extent(std::move(renderer.swapchain_extent))
  { }

  Renderer & operator=(Renderer && renderer){
    instance = std::move(renderer.instance);
#ifdef DEBUG
    messenger = std::move(renderer.messenger);
#endif
    device = std::move(renderer.device);
    surface = std::move(renderer.surface);
    swapchain = std::move(renderer.swapchain);
    swapchain_image_views = std::move(renderer.swapchain_image_views);
    render_pass = std::move(renderer.render_pass);
    descriptor_set_layout = std::move(renderer.descriptor_set_layout);
    pieplien_layout = std::move(renderer.pieplien_layout);
    graphics_pipeline = std::move(renderer.graphics_pipeline);
    depth_image_handles = std::move(renderer.depth_image_handles);
    depth_buffers = std::move(renderer.depth_buffers);
    command_pool = std::move(renderer.command_pool);
    descriptor_pool = std::move(renderer.descriptor_pool);
    descriptor_sets = std::move(renderer.descriptor_sets);
    command_buffers = std::move(renderer.command_buffers);
    vertex_buffer = std::move(renderer.vertex_buffer);
    vertex_buffer_memory = std::move(renderer.vertex_buffer_memory);
    index_buffer = std::move(renderer.index_buffer);
    index_buffer_memory = std::move(renderer.index_buffer_memory);
    uniform_buffers = std::move(renderer.uniform_buffers);
    image_handles = std::move(renderer.image_handles);
    texture_mipmap_levels = std::move(renderer.texture_mipmap_levels);
    texture_image_view = std::move(renderer.texture_image_view);
    texture_sampler = std::move(renderer.texture_sampler);
    swapchain_extent = std::move(renderer.swapchain_extent);

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
  vk::UniqueSurfaceKHR surface;
  vk::UniqueSwapchainKHR swapchain;
  std::vector<vk::UniqueImageView> swapchain_image_views;
  vk::UniqueRenderPass render_pass;
  vk::UniqueDescriptorSetLayout descriptor_set_layout;
  vk::UniquePipelineLayout pieplien_layout;
  vk::UniquePipeline graphics_pipeline;
  Depth_Image_Handles depth_image_handles;
  std::vector<vk::UniqueFramebuffer> depth_buffers;
  vk::UniqueCommandPool command_pool;
  vk::UniqueDescriptorPool descriptor_pool;
  std::vector<vk::UniqueDescriptorSet> descriptor_sets;
  std::vector<vk::UniqueCommandBuffer> command_buffers;
  vk::UniqueBuffer vertex_buffer;
  vk::UniqueDeviceMemory vertex_buffer_memory;
  vk::UniqueBuffer index_buffer;
  vk::UniqueDeviceMemory index_buffer_memory;
  std::vector<std::pair<vk::UniqueBuffer, vk::UniqueDeviceMemory>> uniform_buffers;
  Image_Handles image_handles;
  uint32_t texture_mipmap_levels;
  vk::UniqueImageView texture_image_view;
  vk::UniqueSampler texture_sampler;
  vk::Extent2D swapchain_extent;

  Renderer(Renderer const &) = delete;
  Renderer & operator=(Renderer const &) = delete;
};

#ifdef DEBUG
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    if(messageSeverity & VkDebugUtilsMessageSeverityFlagBitsEXT::VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) spdlog::error("\nvalidation layer: {}\n", pCallbackData->pMessage);
    if(messageSeverity & VkDebugUtilsMessageSeverityFlagBitsEXT::VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) spdlog::warn("\nvalidation layer: {}\n", pCallbackData->pMessage);
    else spdlog::info("\nvalidation layer: {}\n", pCallbackData->pMessage);

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

  renderer.surface = std::invoke([&]{
      VkSurfaceKHR surface;
      if(glfwCreateWindowSurface(*renderer.instance, window, nullptr, &surface)){
        spdlog::error("Unable to create window surface");
        std::abort();
      }

      return vk::UniqueSurfaceKHR(surface, vk::ObjectDestroy<vk::Instance, vk::DispatchLoaderStatic>(renderer.instance.get()));
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

  auto const family_indices = std::invoke([&] noexcept{ 
      struct {
        int graphics_index = -1;
        int present_index = -1;
      } indices; 

      auto const properties = physical_device.getQueueFamilyProperties();

      for(auto i = 0; i < properties.size(); ++i){
        if(indices.graphics_index < 0 and properties[i].queueFlags & vk::QueueFlagBits::eGraphics){
          spdlog::info("Found graphics index {}", 1);
          indices.graphics_index = i;
        }
          
        if(indices.present_index < 0 and physical_device.getSurfaceSupportKHR(1, *renderer.surface)) {
          //spdlog::info("Found present iindex {}", i);
          indices.present_index = i;
        }

        if(indices.graphics_index >= 0 and indices.present_index >= 0){
          return indices;
        }
      }

      spdlog::error("Unable to find graphics card that can display");
      std::abort();
  });

  auto const graphics_family_index = family_indices.graphics_index;
  auto const present_family_index = family_indices.present_index;

  static auto graphicsQueuePriority = 1.0f;

  auto const queueCreateInfos = std::array{
    vk::DeviceQueueCreateInfo{
      .queueFamilyIndex = (uint32_t)graphics_family_index,
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

  auto const graphicsQueue = renderer.device->getQueue(graphics_family_index, 0);

  auto const capabilities = physical_device.getSurfaceCapabilitiesKHR(renderer.surface.get());
  //TODO: pcik a better format
  auto const surface_format = physical_device.getSurfaceFormatsKHR(renderer.surface.get()).back();
  //TODO: pick a better present mode
  auto const present_mode = physical_device.getSurfacePresentModesKHR(renderer.surface.get()).back();

  auto const image_count = capabilities.minImageCount + 1;

  auto const image_format = surface_format.format;
  renderer.swapchain_extent = std::invoke([&]{
      if(capabilities.currentExtent.width not_eq UINT32_MAX)
        return capabilities.currentExtent;

      int width, height;
      glfwGetFramebufferSize(window, &width, &height);

      auto const [
        min_extent_width,
        min_extent_height
      ] = capabilities.minImageExtent;

      auto const [
        max_extent_width,
        max_extent_height
      ] = capabilities.maxImageExtent;

      return vk::Extent2D{
        .width = std::clamp(static_cast<uint32_t>(width), min_extent_width, max_extent_width),
        .height = std::clamp(static_cast<uint32_t>(width), min_extent_height, max_extent_height),
      };
  }); 

  renderer.swapchain = std::invoke([&]{ 
      auto sharing_mode = std::invoke([&]{
        if(graphics_family_index not_eq present_family_index){
          return vk::SharingMode::eExclusive;
        }else{
          return vk::SharingMode::eConcurrent;
        }
      });

      auto info = vk::SwapchainCreateInfoKHR{
        .surface = renderer.surface.get(),
        .minImageCount = image_count,
        .imageFormat = image_format,
        .imageColorSpace = surface_format.colorSpace,
        .imageExtent = renderer.swapchain_extent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = sharing_mode,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = present_mode,
        .clipped = VK_TRUE,
        .oldSwapchain = nullptr,
      };

      return renderer.device->createSwapchainKHRUnique(info);
  });

  auto const create_image_view = [](vk::Device const device, vk::Image const image, vk::Format const format, vk::ImageAspectFlags const aspect_flags, uint32_t mip_levels) noexcept{
    return device.createImageViewUnique(vk::ImageViewCreateInfo{
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = vk::ImageSubresourceRange{
          .aspectMask = aspect_flags,
          .baseMipLevel = 0,
          .levelCount = mip_levels,
          .baseArrayLayer = 0,
          .layerCount = 1,
        }
    });
  };

  renderer.swapchain_image_views = std::invoke([&]{
      auto images = renderer.device->getSwapchainImagesKHR(renderer.swapchain.get());
      auto image_views = std::vector<vk::UniqueImageView>(images.size());
      for(auto i = 0; i < images.size(); ++i){
        image_views[i] = create_image_view(renderer.device.get(), images[i], surface_format.format, vk::ImageAspectFlagBits::eColor, 1);
      } 
      return image_views;
  });

  renderer.render_pass = std::invoke([&] noexcept {
    auto const color_attachment = vk::AttachmentDescription{
      .format = surface_format.format,
      .samples = vk::SampleCountFlagBits::e1,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eStore,
      .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
      .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
      .initialLayout = vk::ImageLayout::eUndefined,
      .finalLayout = vk::ImageLayout::ePresentSrcKHR,
    };

    auto const color_attachment_ref = vk::AttachmentReference{.attachment = 0, .layout = vk::ImageLayout::eColorAttachmentOptimal};

    auto const 

    auto const subpass = vk::SubpassDescription{
      .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
      .colorAttachmentCount = 1,
      .pColorAttachments = & color_attachment_ref,
      .
    };

    return renderer.device->createRenderPass(vk::RenderPassCreateInfo{
        .attachmentCount = 1,
        .pAttachments = &color_attachment,
        .
    });
  });

  return std::move(renderer);
}



void Renderer::draw_frame(){

}
