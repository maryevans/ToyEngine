#pragma once
#include "include.hpp"


struct Queue_Indices{
  uint32_t graphics;
  uint32_t present;
};;

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

//Cube
auto const vertices = std::vector<Vertex>{
    {{0,    0.5,    0.0},     {1,0,1},    {1, 0}},
    {{-0.5, 0,      0.0},     {0,1,0},    {0, 0}},
    {{0.5,  0,      0.0},     {1,0,1},    {0, 1}},
    {{1,    1,      0.0},     {1,1,0},    {1, 1}},

    {{0,    0.5,    -0.5},     {1,0,1},    {1, 0}},
    {{-0.5, 0,      -0.5},     {0,1,0},    {0, 0}},
    {{0.5,  0,      -0.5},     {1,0,1},    {0, 1}},
    {{1,    1,      -0.5},     {1,1,0},    {1, 1}},
};

auto const indices = std::vector<uint32_t>{
    0, 1, 2, 2, 3, 0,
    4, 5, 6, 6, 7, 4
};

uint64_t total_allocated = 0;

constexpr auto validation_layer = "VK_LAYER_KHRONOS_validation";

void * operator new(std::size_t size){

  if(size == 0) ++size;

  if(void * ptr = std::malloc(size)){
    total_allocated += size;
    return ptr;
  };

  spdlog::critical("Bad alloc");
  std::abort();
}

void * operator new[](std::size_t size){
  if(size == 0) ++size;

  if(void * ptr = std::malloc(size)){
    total_allocated += size;
    return ptr;
  };

  spdlog::critical("Bad alloc");
  std::abort();
}

void operator delete(void * ptr, std::size_t size){
  std::free(ptr);
  total_allocated -= size;
}

void operator delete[](void * ptr, std::size_t size){
  std::free(ptr);
  total_allocated -= size;
}

constexpr auto default_viewport(vk::Extent2D swapchain_extent){
  return vk::Viewport{
    .x = 0.0f, .y = 0.0f, 
    .width = static_cast<float>(swapchain_extent.width),
    .height = static_cast<float>(swapchain_extent.height),
    .minDepth = 0.0f,
    .maxDepth = 1.0f,
  };
}

//Creates an object to submit commands to for the current scope.
struct Command_Scope{
    Command_Scope(
        vk::UniqueDevice const & device, 
        vk::UniqueCommandPool const & commandPool, 
        vk::Queue const & graphicsQueue): 
      device(device),
      graphic_queue(graphicsQueue) {

      auto const alloc_info = vk::CommandBufferAllocateInfo{
        .commandPool = commandPool.get(), 
        .level = vk::CommandBufferLevel::ePrimary, 
        .commandBufferCount = 1
      };
      command_buffer = std::move(device->allocateCommandBuffersUnique(alloc_info).back());

      command_buffer->begin(vk::CommandBufferBeginInfo{ 
          .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
      });
    }
    ~Command_Scope(){
        command_buffer->end();

        auto const submit_info = vk::SubmitInfo{
          .commandBufferCount = 1,
          .pCommandBuffers = &command_buffer.get()
        };

        graphic_queue.submit(std::array{submit_info});
    }

    auto & operator ->(){return command_buffer.get();}

    vk::UniqueDevice const & device;
    vk::Queue const & graphic_queue;
    vk::UniqueCommandBuffer command_buffer;
};

class Renderer{
public:
  inline void draw_frame();
  friend inline auto create_renderer(GLFWwindow * window) noexcept;

  inline auto get_viewport()const noexcept{
    return default_viewport(swapchain_extent);
  }

  struct Synchronization{
    vk::UniqueSemaphore image_available_semaphore;
    vk::UniqueSemaphore render_finished_semaphore;
    vk::UniqueFence in_flight_fence;
  };

  vk::UniqueInstance instance;
  vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic> messenger;
  vk::PhysicalDevice physical_device;
  Queue_Indices queue_indices; 
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
  uint32_t max_frames_in_flight = 2;
  std::vector<Renderer::Synchronization> per_frame_sync;
  std::vector<vk::Fence> swapchain_images_in_flight;
  uint32_t current_frame = 0;
};

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    if(messageSeverity & VkDebugUtilsMessageSeverityFlagBitsEXT::VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) spdlog::error("validation layer: {}\n", pCallbackData->pMessage);
    if((messageSeverity & VkDebugUtilsMessageSeverityFlagBitsEXT::VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) == VkDebugUtilsMessageSeverityFlagBitsEXT::VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) 
      spdlog::warn("validation layer: {}\n", pCallbackData->pMessage);
    else spdlog::info("validation layer: {}\n", pCallbackData->pMessage);

  return VK_FALSE;
}


[[nodiscard]]
inline auto create_renderer(GLFWwindow * window) noexcept try{
  spdlog::info("Creating Renderer");

  constexpr auto appinfo = vk::ApplicationInfo{
    .pApplicationName = "Toy",
    .applicationVersion = 0,
    .pEngineName = "Toy",
    .engineVersion = 0,
    .apiVersion = VK_API_VERSION_1_3,
  };

  //TODO: don't allocate here.
  auto layers = std::vector<const char *>(0);
  auto const extensions = std::invoke([]{
    auto glfwExtensionCount = uint32_t(0);
    auto const glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    auto extensions = std::vector<const char *>(glfwExtensions, glfwExtensions + glfwExtensionCount);


    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);


    return extensions;
  });

  for(auto const & layer : vk::enumerateInstanceLayerProperties()){
    if(std::string_view(layer.layerName) == std::string_view(validation_layer)){
      spdlog::info("Found validation layer");
      layers.push_back(validation_layer);
    }
  }

  auto const instanceInfo = vk::InstanceCreateInfo{
    .pApplicationInfo = &appinfo,
    .enabledLayerCount = (uint32_t)layers.size(),
    .ppEnabledLayerNames = layers.data(),
    .enabledExtensionCount = (uint32_t) extensions.size(),
    .ppEnabledExtensionNames = extensions.data(),
  };

  auto instance = vk::createInstanceUnique(instanceInfo);

  auto const messengerInfo = vk::DebugUtilsMessengerCreateInfoEXT{
    .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose,
    .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral 
      | vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
      | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
    .pfnUserCallback = debugCallback,
    .pUserData = nullptr
  };

  auto messenger = instance->createDebugUtilsMessengerEXTUnique(messengerInfo, nullptr, vk::DispatchLoaderDynamic(instance.get(), vkGetInstanceProcAddr));

  auto surface = std::invoke([&]{
      VkSurfaceKHR surface;
      if(glfwCreateWindowSurface(*instance, window, nullptr, &surface)){
        spdlog::error("Unable to create window surface");
        std::abort();
      }

      return vk::UniqueSurfaceKHR(surface, vk::ObjectDestroy<vk::Instance, vk::DispatchLoaderStatic>(instance.get()));
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

  auto const queue_indices = std::invoke([&] noexcept{ 
      struct {
        int graphics_index = -1;
        int present_index = -1;
      } temp_indices; 

      auto const properties = physical_device.getQueueFamilyProperties();

      for(auto i = 0; i < properties.size(); ++i){
        if(temp_indices.graphics_index < 0 and properties[i].queueFlags & vk::QueueFlagBits::eGraphics){
          spdlog::info("Found graphics index {}", 1);
          temp_indices.graphics_index = i;
        }
          
        if(temp_indices.present_index < 0 and physical_device.getSurfaceSupportKHR(1, *surface)) {
          //spdlog::info("Found present iindex {}", i);
          temp_indices.present_index = i;
        }

        if(temp_indices.graphics_index >= 0 and temp_indices.present_index >= 0){
          return Queue_Indices{
            .graphics = (uint32_t)temp_indices.graphics_index,
            .present = (uint32_t)temp_indices.present_index
          };
        }
      }

      spdlog::error("Unable to find graphics card that can display");
      std::abort();
  });

  auto const graphics_family_index = queue_indices.graphics;
  auto const present_family_index = queue_indices.present;

  static auto graphicsQueuePriority = 1.0f;

  auto const queueCreateInfos = std::array{
    vk::DeviceQueueCreateInfo{
      .queueFamilyIndex = (uint32_t)graphics_family_index,
      .queueCount = 1,
      .pQueuePriorities = &graphicsQueuePriority,
    },
  };

  auto const device_extensions = std::array{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  auto device = physical_device.createDeviceUnique(
      vk::DeviceCreateInfo{ 
        .queueCreateInfoCount = queueCreateInfos.size(),
        .pQueueCreateInfos = queueCreateInfos.data(),
        .enabledLayerCount = static_cast<uint32_t>(layers.size()),
        .ppEnabledLayerNames = layers.data(),
        .enabledExtensionCount = device_extensions.size(),
        .ppEnabledExtensionNames = device_extensions.data()
      });

  auto const graphicsQueue = device->getQueue(graphics_family_index, 0);

  auto const capabilities = physical_device.getSurfaceCapabilitiesKHR(surface.get());
  //TODO: pcik a better format
  auto const surface_format = physical_device.getSurfaceFormatsKHR(surface.get()).back();
  //TODO: pick a better present mode
  auto const present_mode = physical_device.getSurfacePresentModesKHR(surface.get()).back();

  auto const image_count = capabilities.minImageCount + 1;

  auto const image_format = surface_format.format;
  auto swapchain_extent = std::invoke([&]{
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

  auto swapchain = std::invoke([&]{ 
      auto sharing_mode = std::invoke([&]{
        if(graphics_family_index not_eq present_family_index){
          return vk::SharingMode::eExclusive;
        }else{
          return vk::SharingMode::eConcurrent;
        }
      });

      auto info = vk::SwapchainCreateInfoKHR{
        .surface = surface.get(),
        .minImageCount = image_count,
        .imageFormat = image_format,
        .imageColorSpace = surface_format.colorSpace,
        .imageExtent = swapchain_extent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = sharing_mode,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = present_mode,
        .clipped = VK_TRUE,
        .oldSwapchain = nullptr,
      };

      return device->createSwapchainKHRUnique(info);
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

  auto swapchain_image_views = std::invoke([&]{
      auto images = device->getSwapchainImagesKHR(swapchain.get());
      auto image_views = std::vector<vk::UniqueImageView>(images.size());
      for(auto i = 0; i < images.size(); ++i){
        image_views[i] = create_image_view(device.get(), images[i], surface_format.format, vk::ImageAspectFlagBits::eColor, 1);
      } 
      return image_views;
  });

  auto render_pass = std::invoke([&] noexcept {
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

    return device->createRenderPassUnique(info);
  });

  auto descriptor_set_layout = std::invoke([&]{

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
      
      return device->createDescriptorSetLayoutUnique(info);
  });

  auto graphics_pipeline_layout = std::invoke([&]{

      auto const info = vk::PipelineLayoutCreateInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout.get()
      };

      return device->createPipelineLayoutUnique(info);
  });

  auto load_shader_module = [&](std::filesystem::path shader){
    auto file = std::ifstream(shader.string(), std::ios::ate | std::ios::binary);
    auto error = std::error_code();
    auto const file_size = std::filesystem::file_size(shader, error);
    if(error){
      spdlog::critical("Unable to read size of shader file. {}", error.message());
      std::abort();
    }
    auto buffer = std::vector<char>(file_size);
    file.seekg(0);
    file.read(buffer.data(), file_size);

    auto const shader_info = vk::ShaderModuleCreateInfo{
      .codeSize = buffer.size(),
      .pCode = reinterpret_cast<uint32_t const *>(buffer.data())
    };

    return device->createShaderModuleUnique(shader_info);
  };

  spdlog::info("Loading shader modules");
  auto vert_shader = load_shader_module("./vert.spv");
  spdlog::info("Loaded vert shader");
  auto frag_shader = load_shader_module("./frag.spv");
  spdlog::info("Loaded frag shader");

  auto graphics_pipeline = std::invoke([&] {
    spdlog::info("Createing render pipeline");

    spdlog::trace("create vert shader stage info");
    auto const vert_shader_stage = vk::PipelineShaderStageCreateInfo{
      .stage = vk::ShaderStageFlagBits::eVertex,
      .module = vert_shader.get(),
      .pName = "main"
    };

    spdlog::trace("create frag shader stage info");
    auto const frag_shader_stage  = vk::PipelineShaderStageCreateInfo{
      .stage = vk::ShaderStageFlagBits::eFragment,
      .module = frag_shader.get(),
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
    auto const viewport = default_viewport(swapchain_extent);

    auto const scissor = vk::Rect2D{
      .offset = {0,0},
      .extent = swapchain_extent
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
      .pInputAssemblyState = &input_assembly_state,
      .pViewportState = &viewport_state,
      .pRasterizationState = &rasterization_state,
      .pMultisampleState = &multisampling,
      .pDepthStencilState = &depth_stencil,
      .pColorBlendState = &color_blending,
      .pDynamicState = &dynamic_state,
      .layout = graphics_pipeline_layout.get(),
      .renderPass = render_pass.get()
    };

    spdlog::trace("Creating graphics pipeline");
    auto result = device->createGraphicsPipelineUnique({}, info);

    if(static_cast<VkResult>(result.result) not_eq VK_SUCCESS){
      spdlog::critical("Unable to create graphics pipeline");
      std::abort();
    }

    return std::move(result.value);
  }); 

  spdlog::info("Setting up frame buffers");
  auto frame_buffers = std::invoke([&]{
    auto frame_buffers = std::vector<vk::UniqueFramebuffer>();
    frame_buffers.reserve(swapchain_image_views.size());
    for(auto const & imageView : swapchain_image_views){
        
        auto const attachments = std::vector{ 
            imageView.get(), 
            //depthImageView.get() 
        };

        auto const info = vk::FramebufferCreateInfo{
          .renderPass = render_pass.get(),
          .attachmentCount = static_cast<uint32_t>(attachments.size()),
          .pAttachments = attachments.data(),
          .width = swapchain_extent.width,
          .height = swapchain_extent.height,
          .layers = 1
        };

        frame_buffers.push_back(device->createFramebufferUnique(info));
    }
    return frame_buffers;
  });

  auto command_pool = device->createCommandPoolUnique(vk::CommandPoolCreateInfo{
      .queueFamilyIndex = static_cast<uint32_t>(graphics_family_index)
  });

  spdlog::info("Setting up descriptor pools");
  auto descriptor_pool = std::invoke([&]{
      auto const uniform_buffer_pool_size = vk::DescriptorPoolSize{
        .type = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = (uint32_t)swapchain_image_views.size()
      };

      auto const texture_sampler_pool_size = vk::DescriptorPoolSize{
        .type = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = (uint32_t)swapchain_image_views.size()
      };

      auto const pool_sizes = std::array{uniform_buffer_pool_size, texture_sampler_pool_size};

      auto const pool_info = vk::DescriptorPoolCreateInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .poolSizeCount = pool_sizes.size(),
        .pPoolSizes = pool_sizes.data()
      };

      return device->createDescriptorPoolUnique(pool_info);
  });

  auto const graphics_queue = device->getQueue(graphics_family_index, 0);

  auto const find_memory_type_index = [&](vk::MemoryRequirements requirements, vk::MemoryPropertyFlags memory_flags)->uint32_t{
      auto const memory_properties = physical_device.getMemoryProperties();
    for(auto memory_index = 0; memory_index < memory_properties.memoryTypeCount; ++memory_index){
      if(requirements.memoryTypeBits & (1 << memory_index) && (memory_properties.memoryTypes[memory_index].propertyFlags & memory_flags )){
        return memory_index;
      }
    }

      throw std::runtime_error("Unable to find memory type");
  };

  auto const create_buffer = [&](
      vk::DeviceSize size,
      vk::BufferUsageFlags usage, 
      vk::MemoryPropertyFlags memory_properties)
  {
    auto buffer_info = vk::BufferCreateInfo{
      .size = size,
      .usage = usage,
      .sharingMode = vk::SharingMode::eExclusive
    };
    auto buffer = device->createBufferUnique(buffer_info);
    auto const buffer_memory_requirements = device->getBufferMemoryRequirements(buffer.get());
    auto buffer_memory_info = vk::MemoryAllocateInfo{
      .allocationSize = buffer_memory_requirements.size,
      .memoryTypeIndex = find_memory_type_index(buffer_memory_requirements, memory_properties), 
    };
    auto buffer_memory = device->allocateMemoryUnique(buffer_memory_info);
    device->bindBufferMemory(buffer.get(), buffer_memory.get(), 0);

    struct{
      vk::UniqueBuffer buffer;
      vk::UniqueDeviceMemory buffer_memory;
    } buffer_handles;
    buffer_handles.buffer = std::move(buffer);
    buffer_handles.buffer_memory = std::move(buffer_memory);
    return std::move(buffer_handles);
  };

  auto const copy_buffer = [&](vk::UniqueBuffer const & src_buffer, vk::UniqueBuffer const & dst_buffer, vk::DeviceSize size){
    auto command_scope = Command_Scope(device, command_pool, graphics_queue);
    command_scope.command_buffer->copyBuffer(
        *src_buffer, 
        *dst_buffer, 
        {vk::BufferCopy{.size = size }});
  };

  spdlog::info("Setting up vertex buffers");
  auto vertex_buffer_handles = std::invoke([&]{

    auto const buffer_size = vk::DeviceSize(sizeof(Vertex) * vertices.size());

    auto host_memory_flag_bits = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

    auto host_buffer_handles = create_buffer(
        buffer_size, 
        vk::BufferUsageFlagBits::eTransferSrc, 
        host_memory_flag_bits
    );

    auto vertex_buffer_handles = create_buffer(
        buffer_size,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, 
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    copy_buffer(host_buffer_handles.buffer, vertex_buffer_handles.buffer, buffer_size);

    return vertex_buffer_handles;
  });

  spdlog::info("Settung up index buffers");
  auto index_buffer_handles = std::invoke([&]{
    auto vertices = std::vector{
      Vertex{.position={0,0,0}},
      {.position={1,0,0}},
      {.position={.5,1,0}}
    };

    auto const buffer_size = vk::DeviceSize(sizeof(Vertex) * vertices.size());

    auto host_memory_flag_bits = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

    auto host_buffer_handles = create_buffer(
        buffer_size, 
        vk::BufferUsageFlagBits::eTransferSrc, 
        host_memory_flag_bits
    );

    auto vertex_buffer_handles = create_buffer(
        buffer_size,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, 
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    copy_buffer(host_buffer_handles.buffer, vertex_buffer_handles.buffer, buffer_size);

    return vertex_buffer_handles;
  });


  auto const command_buffer_info = vk::CommandBufferAllocateInfo{
    .commandPool = command_pool.get(),
    .commandBufferCount =(uint32_t)frame_buffers.size() 
  };
  auto command_buffers = device->allocateCommandBuffersUnique(command_buffer_info);

  spdlog::info("Creating command buffers per fram");
  for(auto i = 0; i < frame_buffers.size(); ++i){
    auto const & commandBuffer = command_buffers[i];
    auto const & frame_buffer = frame_buffers[i];

    commandBuffer->begin(vk::CommandBufferBeginInfo{});

    auto const clearColor = std::vector{
        vk::ClearValue {
          vk::ClearColorValue {
            std::array{0.0f, 0.0f, 0.0f ,0.0f}
          }
        },
//        vk::ClearValue{.depthStencil = {1.0f, 0}}
    };

    auto const renderArea = vk::Rect2D{
      {0,0}, 
      swapchain_extent
    };

    auto const render_pass_info = vk::RenderPassBeginInfo{
        .renderPass = render_pass.get(), 
        .framebuffer = frame_buffer.get(), 
        .renderArea = renderArea, 
        .clearValueCount = (uint32_t)clearColor.size(),
        .pClearValues = clearColor.data()
      };

    spdlog::trace("adding render pass to command buffer {}", i);
    commandBuffer->beginRenderPass(render_pass_info, vk::SubpassContents::eInline);

    spdlog::trace("setting viewport for command buffer {}", i);
    commandBuffer->setViewport(0, default_viewport(swapchain_extent));

    spdlog::trace("setting scissor for command buffer {}", i);
    auto const scissor = vk::Rect2D{{0,0}, swapchain_extent};
    commandBuffer->setScissor(0, 1, &scissor);

    commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, graphics_pipeline.get());

    vk::DeviceSize offsets[] = {0};
    spdlog::trace("binding vertex and index buffers to command buffer {}", i);
    commandBuffer->bindVertexBuffers(0, 1, &vertex_buffer_handles.buffer.get(), offsets);
    commandBuffer->bindIndexBuffer(index_buffer_handles.buffer.get(), 0, vk::IndexType::eUint32);

    //commandBuffer->bindDescriptorSets(
    //  vk::PipelineBindPoint::eGraphics, 
    //  graphics_pipeline_layout.get(), 
    //  0, 
    //  0,
    //  nullptr, 
    //  0, 
    //  nullptr
    //);

    //commandBuffer->draw(static_cast<uint32_t>(vertices.size()),1,0,0);
    commandBuffer->drawIndexed(indices.size(), 1, 0, 0, 0);
    spdlog::trace("finnishing up command buffer {}", i);
    commandBuffer->endRenderPass();
    commandBuffer->end();
  }


  auto const create_synchronization = [&](uint32_t max_frames_in_flight){
    auto per_frame_sync = std::vector<Renderer::Synchronization>(max_frames_in_flight);
    for(auto && frame_sync: per_frame_sync){
      frame_sync = Renderer::Synchronization{
        .image_available_semaphore = device->createSemaphoreUnique(vk::SemaphoreCreateInfo{}),
        .render_finished_semaphore = device->createSemaphoreUnique(vk::SemaphoreCreateInfo{}),
        .in_flight_fence = device->createFenceUnique(vk::FenceCreateInfo{
          .flags = vk::FenceCreateFlagBits::eSignaled
        })
      };
    }
    return per_frame_sync;
  };

  auto const create_swapchain_images_in_flight_fences = [&]{
    auto fences = std::vector<vk::Fence>();
    fences.reserve(swapchain_image_views.size());
    for(auto i =0; i < swapchain_image_views.size(); ++i){
      fences[i] = device->createFence(vk::FenceCreateInfo{});
    }
    return fences;
  };

  constexpr int max_frames_in_flight = 2;

  spdlog::info("Createing Renderer object");

  return Renderer{
    .instance = std::move(instance),
    .messenger = std::move(messenger),
    .physical_device = physical_device,
    .queue_indices = queue_indices, 
    .device = std::move(device),
    .surface = std::move(surface),
    .swapchain = std::move(swapchain),
    .swapchain_image_views = std::move(swapchain_image_views),
    .render_pass = std::move(render_pass),
    .descriptor_set_layout = std::move(descriptor_set_layout),
    .vert_shader = std::move(vert_shader),
    .frag_shader = std::move(frag_shader),
    .graphics_pipeline_layout = std::move(graphics_pipeline_layout),
    .graphics_pipeline = std::move(graphics_pipeline),
    //.depth_image_handles = std::move(depth_image_handles),
    //.depth_buffers = std::move(depth_buffers),
    .command_pool = std::move(command_pool),
    //.descriptor_pool = std::move(descriptor_pool),
    //.descriptor_sets = std::move(descriptor_sets),
    .command_buffers = std::move(command_buffers),
    .vertex_buffer = std::move(vertex_buffer_handles.buffer),
    .vertex_buffer_memory = std::move(vertex_buffer_handles.buffer_memory),
    .index_buffer = std::move(index_buffer_handles.buffer),
    .index_buffer_memory = std::move(index_buffer_handles.buffer_memory),
    //.uniform_buffers = std::move(uniform_buffers),
    //.image_handles = std::move(image_handles),
    //.texture_mipmap_levels = std::move(texture_mipmap_levels),
    //.texture_image_view = std::move(texture_image_view),
    //.texture_sampler = std::move(texture_sampler),
    .swapchain_extent = std::move(swapchain_extent),
    .max_frames_in_flight = max_frames_in_flight,
    .per_frame_sync = create_synchronization(max_frames_in_flight),
    .swapchain_images_in_flight = create_swapchain_images_in_flight_fences(),
  };
} catch (std::exception & exception){
  spdlog::critical("Exception:{}", exception.what());
  std::abort();
}

void Renderer::draw_frame(){
  auto const & [
     image_available_semaphore,
     render_finished_semaphore,
     in_flight_fence
  ] = per_frame_sync[current_frame];

  if(device->waitForFences(1, &in_flight_fence.get(), VK_TRUE, UINT64_MAX) != vk::Result::eSuccess){
    spdlog::warn("Unable to wait for current frame {}", current_frame);
  }
  
  auto const next_image_result = device->acquireNextImageKHR(
    swapchain.get(), 
    UINT64_MAX, 
    image_available_semaphore.get(), 
    in_flight_fence.get()
  );

  //TODO: don't crash
  if(next_image_result.result not_eq vk::Result::eSuccess){
    throw std::runtime_error("failed to present swapchain image.");
  }

  auto image_index = next_image_result.value;
  
  //if(imageIndex.result == vk::Result::eErrorOutOfDateKHR or imageIndex.result == vk::Result::eSuboptimalKHR or frameResized){
  //    renderState = createVulkanRenderState(device, gpu, surface, window, graphicsIndex, presentIndex);
  //    frameResized = false;
  //    return;
  //}
  //
  auto & image_in_flight = swapchain_images_in_flight[image_index];
  
  if(image_in_flight){
    if(device->waitForFences(1, &image_in_flight, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess){
      std::cout << "Unable to wait for image in flight fence: " << image_index << std::endl;
    }
  }

  //TODO: instead two different indapendendet sets of fences should exist for sync.
  image_in_flight = *in_flight_fence;
  
  
  //auto const & buffer_handles = uniform_buffers[image_index.value];

  
  //update_uniformBuffer(
  //        device.get(), 
  //        bufferHandles.first.get(), 
  //        bufferHandles.second.get(), 
  //        renderState->swapchainExtent);
  
  
  vk::Semaphore wait_semaphores[] = {image_available_semaphore.get()};
  vk::PipelineStageFlags wait_dst_stage_masks[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
  vk::Semaphore signal_semaphores[] = {render_finished_semaphore.get()};
  vk::CommandBuffer graphics_queue_command_buffers[] = {command_buffers[image_index].get()};
  auto const submit_info = vk::SubmitInfo{
    .waitSemaphoreCount = 1, .pWaitSemaphores = wait_semaphores,
    .pWaitDstStageMask = wait_dst_stage_masks,
    .commandBufferCount = 1, .pCommandBuffers = graphics_queue_command_buffers,
    .signalSemaphoreCount = 1, .pSignalSemaphores = signal_semaphores
  };
  
  if(device->resetFences(1, &per_frame_sync[current_frame].in_flight_fence.get()) != vk::Result::eSuccess){
    spdlog::warn("Unable to reset fence: {}", current_frame);
  }
  
  auto graphics_queue = device->getQueue(queue_indices.graphics, 0);

  if(graphics_queue.submit(1, &submit_info, in_flight_fence.get()) != vk::Result::eSuccess){
    spdlog::warn("Bad submit");
  }
  
  auto const present_info = vk::PresentInfoKHR{
    .waitSemaphoreCount = 1, .pWaitSemaphores = signal_semaphores,
    .swapchainCount = 1, .pSwapchains = &swapchain.get(),
    .pImageIndices = &image_index
  };

  auto present_queue = device->getQueue(queue_indices.present, 0);
  
  if(present_queue.presentKHR(present_info) != vk::Result::eSuccess){
    spdlog::warn("Bad present");
  }
  
  present_queue.waitIdle();
  
  current_frame = (current_frame + 1) % max_frames_in_flight;
}
