#pragma once
#define DEBUG
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

struct Vertex{
    glm::vec3 position;
    glm::vec3 color;
    glm::vec2 tex_coord;
};

uint64_t total_allocated = 0;

void * operator new(std::size_t size){

  if(size == 0) ++size;

  if(void * ptr = std::malloc(size)){
    total_allocated += size;
    //fmt::print("Allocated bytes {}, total {}\n", size, total_allocated);
    return ptr;
  };

  spdlog::critical("Bad alloc");
  std::abort();
}

void * operator new[](std::size_t size){
  if(size == 0) ++size;

  if(void * ptr = std::malloc(size)){
    total_allocated += size;
    //fmt::print("Allocated bytes {}, total {}\n", size, total_allocated);
    return ptr;
  };

  spdlog::critical("Bad alloc");
  std::abort();
}

void operator delete(void * ptr, std::size_t size){
  std::free(ptr);
  total_allocated -= size;
  fmt::print("Deleted {} bytes, total {}\n", size, total_allocated);
}

void operator delete[](void * ptr, std::size_t size){
  std::free(ptr);
  total_allocated -= size;
  fmt::print("Deleted {} bytes, total {}\n", size, total_allocated);
}

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
    vert_shader(std::move(renderer.vert_shader)),
    frag_shader(std::move(renderer.frag_shader)),
    graphics_pipeline_layout(std::move(renderer.graphics_pipeline_layout)),
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
    vert_shader = std::move(renderer.vert_shader);
    frag_shader = std::move(renderer.frag_shader);
    graphics_pipeline_layout = std::move(renderer.graphics_pipeline_layout);
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
  vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic> messenger;
#endif
  vk::UniqueDevice device;
  vk::UniqueSurfaceKHR surface;
  vk::UniqueSwapchainKHR swapchain;
  std::vector<vk::UniqueImageView> swapchain_image_views;
  vk::UniqueRenderPass render_pass;
  vk::UniqueDescriptorSetLayout descriptor_set_layout;
  vk::UniqueShaderModule vert_shader;
  vk::UniqueShaderModule frag_shader;
  vk::UniquePipelineLayout graphics_pipeline_layout;
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
    if((messageSeverity & VkDebugUtilsMessageSeverityFlagBitsEXT::VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) == VkDebugUtilsMessageSeverityFlagBitsEXT::VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) 
      spdlog::warn("\nvalidation layer: {}\n", pCallbackData->pMessage);
    else spdlog::info("\nvalidation layer: {}\n", pCallbackData->pMessage);

    return VK_FALSE;
}
#endif

[[nodiscard]]
inline auto create_renderer(GLFWwindow * window) noexcept try{
  spdlog::info("Creating Renderer");

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

  auto const layers = std::invoke([&] -> std::vector<const char *>{
#ifdef DEBUG
    return std::vector{"VK_LAYER_KHRONOS_validation"};
#else
    return std::vector<const char*>{};
#endif
  });

  {
      auto found_layers = std::vector<bool>(layers.size());

      for(auto const & layer : vk::enumerateInstanceLayerProperties()){
        spdlog::info("Layer {}", layer.layerName);
        for(auto i =0; i < layers.size(); ++i){
          if(std::string_view(layer.layerName) == std::string_view(layers[i])){
            spdlog::info("Found Layer {}", layer.layerName);
            found_layers[i] = true;
          }
        }
      }

      for(auto i = 0; i < layers.size(); ++i){
        spdlog::info("bla {}" ,found_layers[i]);
        if(not found_layers[i]) spdlog::critical("Missing vulkan layer: {}", layers[i]);
      }

      for(auto const & layer : found_layers) if(not layer) std::abort();
  }

  auto const instanceInfo = vk::InstanceCreateInfo{
    .pApplicationInfo = &appinfo,
    .enabledLayerCount = static_cast<uint32_t>(layers.size()),
    .ppEnabledLayerNames = layers.data(),
    .enabledExtensionCount = (uint32_t) extensions.size(),
    .ppEnabledExtensionNames = extensions.data(),
  };

  renderer.instance = vk::createInstanceUnique(instanceInfo);

#ifdef DEBUG
  auto const messengerInfo = vk::DebugUtilsMessengerCreateInfoEXT{
    .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose,
    .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral 
      | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation,
    .pfnUserCallback = debugCallback,
    .pUserData = nullptr
  };

  renderer.messenger = renderer.instance->createDebugUtilsMessengerEXTUnique(messengerInfo, nullptr, vk::DispatchLoaderDynamic(renderer.instance.get(), vkGetInstanceProcAddr));
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
        .enabledLayerCount = static_cast<uint32_t>(layers.size()),
        .ppEnabledLayerNames = layers.data(),
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


    auto const color_attachment_ref = vk::AttachmentReference{
    .attachment = 0, 
    .layout = vk::ImageLayout::eColorAttachmentOptimal
    };


    auto const depth_format = std::invoke([&]{
        //TODO: figure out what formats are needed for depth
        auto const formats ={
          vk::Format::eD32Sfloat,
          vk::Format::eD32SfloatS8Uint,
          vk::Format::eD24UnormS8Uint
        };

        auto const feature = vk::FormatFeatureFlagBits::eDepthStencilAttachment;

        for(auto const & format : formats){
          auto const props = physical_device.getFormatProperties(format);
          if((props.optimalTilingFeatures & feature) == feature){
            return format;
          }
        }

        spdlog::error("Unable to get depth format");
        std::abort();
    });

    auto const depth_attachment = vk::AttachmentDescription{
      .format = depth_format,
      .samples = vk::SampleCountFlagBits::e1,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eStore,
      .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
      .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
      .initialLayout = vk::ImageLayout::eUndefined,
      .finalLayout = vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal
    };

    auto const depth_attachment_ref = vk::AttachmentReference{
      .attachment = 1,
      .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
    };

    auto const subpass = vk::SubpassDescription{
      .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
      .colorAttachmentCount = 1,
      .pColorAttachments = & color_attachment_ref,
      .pDepthStencilAttachment = & depth_attachment_ref,
    };

    auto const subpass_dependency = vk::SubpassDependency{
      .srcSubpass = VK_SUBPASS_EXTERNAL,
      .dstSubpass = 0,
      .srcStageMask = 
        vk::PipelineStageFlagBits::eColorAttachmentOutput 
        | vk::PipelineStageFlagBits::eEarlyFragmentTests,
      .dstStageMask =
        vk::PipelineStageFlagBits::eColorAttachmentOutput
        | vk::PipelineStageFlagBits::eEarlyFragmentTests,
      .srcAccessMask = {},
      .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite
        | vk::AccessFlagBits::eDepthStencilAttachmentWrite
    };

    auto const attachments = std::array{color_attachment, depth_attachment};
    auto const info = vk::RenderPassCreateInfo{
      .flags = {},
      .attachmentCount = attachments.size(),
      .pAttachments = attachments.data(),
      .subpassCount = 1,
      .pSubpasses = &subpass,
      .dependencyCount = 1,
      .pDependencies = &subpass_dependency,
    };

    return renderer.device->createRenderPassUnique(info);
  });

  renderer.descriptor_set_layout = std::invoke([&]{

      auto const uniform_buffer_binding = vk::DescriptorSetLayoutBinding{
        .binding = 0,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eVertex
      };

      auto const sampler_binding = vk::DescriptorSetLayoutBinding{
        .binding = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
      };

      auto const descriptor_sets = std::array{uniform_buffer_binding, sampler_binding};

      auto info = vk::DescriptorSetLayoutCreateInfo{
        .bindingCount = descriptor_sets.size(),
        .pBindings = descriptor_sets.data()
      };
      
      return renderer.device->createDescriptorSetLayoutUnique(info);
  });

  renderer.graphics_pipeline_layout = std::invoke([&]{

      auto const info = vk::PipelineLayoutCreateInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &renderer.descriptor_set_layout.get()
      };

      return renderer.device->createPipelineLayoutUnique(info);
  });

  auto load_shader_module = [&](std::filesystem::path shader){
    auto file = std::ifstream(shader.string(), std::ios::ate | std::ios::binary);
    auto const fileSize = (size_t) file.tellg();
    auto buffer = std::vector<char>(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);

    auto const shader_info = vk::ShaderModuleCreateInfo{
      .codeSize = buffer.size(),
      .pCode = reinterpret_cast<uint32_t const *>(buffer.data())
    };

    return renderer.device->createShaderModuleUnique(shader_info);
  };

  renderer.vert_shader = load_shader_module("./vert.spv");
  renderer.frag_shader = load_shader_module("./frag.spv");

  renderer.graphics_pipeline = std::invoke([&] {
    spdlog::info("Createing render pipeline");

    spdlog::trace("create vert shader stage info");
    auto const vert_shader_stage = vk::PipelineShaderStageCreateInfo{
      .stage = vk::ShaderStageFlagBits::eVertex,
      .module = renderer.vert_shader.get(),
      .pName = "main"
    };

    spdlog::trace("create frag shader stage info");
    auto const frag_shader_stage  = vk::PipelineShaderStageCreateInfo{
      .stage = vk::ShaderStageFlagBits::eFragment,
      .module = renderer.frag_shader.get(),
      .pName = "main"
    };

    auto const shader_stages = std::array{vert_shader_stage, frag_shader_stage};

    spdlog::trace("creating vertex binding description");
    //TODO: make this bind an structure of arrays.
    auto const vertex_binding_description = vk::VertexInputBindingDescription{
      .binding=0,
      .stride = sizeof(Vertex),
      .inputRate = vk::VertexInputRate::eVertex
    };

    spdlog::trace("creating vertex binding attributes");
    auto const vertex_binding_attributes = std::array{
      vk::VertexInputAttributeDescription{
        .location = 0,
        .binding = vertex_binding_description.binding,
        .format = vk::Format::eR32G32B32Sfloat,
        .offset = offsetof(Vertex, position),
      },
      vk::VertexInputAttributeDescription{
        .location = 1,
        .binding = vertex_binding_description.binding,
        .format = vk::Format::eR32G32B32Sfloat,
        .offset = offsetof(Vertex, color)
      },
      vk::VertexInputAttributeDescription{
        .location = 2,
        .binding = vertex_binding_description.binding,
        .format = vk::Format::eR32G32Sfloat,
        .offset = offsetof(Vertex, tex_coord)
      }
    };

    spdlog::trace("creating vertex input state info");
    auto const vertex_input_state = vk::PipelineVertexInputStateCreateInfo{
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vertex_binding_description,
      .vertexAttributeDescriptionCount = vertex_binding_attributes.size(),
      .pVertexAttributeDescriptions = vertex_binding_attributes.data(),
    };

    spdlog::trace("creating input assembly state info");
    auto const input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo{
      .topology = vk::PrimitiveTopology::eTriangleList,
      .primitiveRestartEnable = VK_FALSE
    };

    spdlog::trace("defining viewport");
    auto const viewport = vk::Viewport{
      .x = 0.0f, .y = 0.0f, 
      .width = static_cast<float>(renderer.swapchain_extent.width),
      .height = static_cast<float>(renderer.swapchain_extent.height),
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
    };

    auto const scissor = vk::Rect2D{
      .offset = {0,0},
      .extent = renderer.swapchain_extent
    };

    spdlog::trace("creating viewport state info");
    auto const viewport_state = vk::PipelineViewportStateCreateInfo{
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor,
    };

    auto const rasterization_state = vk::PipelineRasterizationStateCreateInfo{
      .depthClampEnable = VK_FALSE,
      .rasterizerDiscardEnable = VK_FALSE,
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = vk::CullModeFlagBits::eBack,
      .frontFace = vk::FrontFace::eCounterClockwise,
      .depthBiasEnable = VK_FALSE,
    };

    auto const multisampling = vk::PipelineMultisampleStateCreateInfo{
      .rasterizationSamples = vk::SampleCountFlagBits::e1,
      .sampleShadingEnable = VK_FALSE,
      .minSampleShading = 1.0f,
      .pSampleMask = nullptr,
      .alphaToCoverageEnable = VK_FALSE,
      .alphaToOneEnable = VK_FALSE,
    };

    auto const color_blend_attachemnt = vk::PipelineColorBlendAttachmentState{
      .blendEnable = VK_FALSE,
      .srcColorBlendFactor = vk::BlendFactor::eOne,
      .dstAlphaBlendFactor = vk::BlendFactor::eOne,
      .colorWriteMask = vk::ColorComponentFlagBits::eR
        | vk::ColorComponentFlagBits::eG
        | vk::ColorComponentFlagBits::eB
        | vk::ColorComponentFlagBits::eA
    };

    auto const blend_constants = std::array{0.0f, 0.0f, 0.0f, 0.0f};

    auto const color_blending = vk::PipelineColorBlendStateCreateInfo{
      .logicOpEnable = VK_FALSE,
      .logicOp = vk::LogicOp::eCopy,
      .attachmentCount = 1,
      .pAttachments = &color_blend_attachemnt,
      .blendConstants = blend_constants
    };

    auto const dynamic_states = std::array{
      vk::DynamicState::eViewport,
      vk::DynamicState::eScissor,
      vk::DynamicState::eLineWidth
    };

    auto const dynamic_state = vk::PipelineDynamicStateCreateInfo{
      .dynamicStateCount = dynamic_states.size(),
      .pDynamicStates = dynamic_states.data()
    };

    auto const depth_stencil = vk::PipelineDepthStencilStateCreateInfo{
      .depthTestEnable = VK_TRUE,
      .depthWriteEnable = VK_TRUE,
      .depthCompareOp = vk::CompareOp::eLess,
      .depthBoundsTestEnable = VK_FALSE,
      .stencilTestEnable = VK_FALSE,
      .minDepthBounds = 0.0f,
      .maxDepthBounds = 1.0f
    };

    auto const info = vk::GraphicsPipelineCreateInfo{
      .stageCount = shader_stages.size(), 
      .pStages = shader_stages.data(),
      .pVertexInputState = &vertex_input_state,
      .pViewportState = &viewport_state,
      .pRasterizationState = &rasterization_state,
      .pMultisampleState = &multisampling,
      .pDepthStencilState = &depth_stencil,
      .pColorBlendState = &color_blending,
      .pDynamicState = &dynamic_state,
      .layout = renderer.graphics_pipeline_layout.get(),
      .renderPass = renderer.render_pass.get()
    };

    spdlog::trace("Creating graphics pipeline");
    auto result = renderer.device->createGraphicsPipelineUnique({}, info);

    if(static_cast<VkResult>(result.result) not_eq VK_SUCCESS){
      spdlog::critical("Unable to create graphics pipeline");
      throw std::runtime_error("bla");
    }

    spdlog::trace("jlajf");
    return std::move(result.value);
  }); 

  spdlog::trace("jlajf");
  return std::move(renderer);
} catch (std::exception & exception){
  spdlog::critical("Exception:{}", exception.what());
  std::abort();
}



void Renderer::draw_frame(){

}
