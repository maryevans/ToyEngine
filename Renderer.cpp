#pragma once
#include "include.hpp"

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);
struct vulkan_data{
  VkInstance instance;
  VkDebugUtilsMessengerEXT debug_messenger;
  VkDevice device;
  VkQueue present_queue;

  VkSurfaceKHR surface;
  VkSwapchainKHR swapchain;

  VkFormat format;
  VkExtent2D extent;

  VkPipelineLayout pipeline_layout;

  VkRenderPass render_pass;
  // std::vector<VkImage> swapchain_images;
  std::vector<VkImageView> swapchain_image_views;

  VkPipeline graphics_pipeline;

  VkCommandPool command_pool;
  VkCommandBuffer command_buffer;
  std::vector<VkFramebuffer> swapchain_framebuffers;
  ~vulkan_data(){
    spdlog::info("In destructor");
    vkDestroyCommandPool(device, command_pool, nullptr);
    for(auto framebuffer : swapchain_framebuffers){
      vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    vkDestroyPipeline(device, graphics_pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyRenderPass(device, render_pass, nullptr);
    for(auto image_view : swapchain_image_views){
      vkDestroyImageView(device, image_view, nullptr);
    }
    vkDestroySwapchainKHR(device, swapchain, nullptr);
    vkDestroyDevice(device, nullptr);
    DestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
  }
};

struct Queue_Indices{
  uint32_t graphics;
  uint32_t present;
};


auto check_validation(){
  uint32_t layer_count;
  vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
  VkLayerProperties available_layers[layer_count];
  vkEnumerateInstanceLayerProperties(&layer_count, available_layers);

  const char* validation_layers[] = {"VK_LAYER_KHRONOS_validation"};
  bool layer_found = false;
  for(auto const v_layer : validation_layers){
    
    for(auto const& layer_prop : available_layers){
      if(v_layer == layer_prop.layerName){
        layer_found = true;
        break;
      }
    }
    if(layer_found){break;}
  }
  return layer_found;

}


static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

  if((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) == VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT){
    spdlog::error(pCallbackData->pMessage);
  }else if((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) == VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT){
    spdlog::warn(pCallbackData->pMessage);
  }else if((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) == VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT){
    spdlog::info(pCallbackData->pMessage);
  }else{
    spdlog::trace(pCallbackData->pMessage);
  }
  return VK_FALSE;
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}


auto create_renderer(GLFWwindow * window){
  const char* validation_layers[] = {"VK_LAYER_KHRONOS_validation"};


  vulkan_data vk_data;

  auto app_info = VkApplicationInfo{
    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    .pApplicationName = "Toy",
    .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
    .pEngineName = "Toy",
    .engineVersion = VK_MAKE_VERSION(1, 0, 0),
    .apiVersion = VK_API_VERSION_1_3
  };

  uint32_t glfw_extension_count = 0;
  auto const glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
  
  auto extensions = std::vector<const char*>(glfw_extensions, glfw_extensions + glfw_extension_count);
  extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  auto instance_info = VkInstanceCreateInfo{
    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    .pApplicationInfo = &app_info,
    .enabledLayerCount = 1,
    .ppEnabledLayerNames = validation_layers,
    .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
    .ppEnabledExtensionNames = extensions.data()
  };

  VkInstance instance;

  if(vkCreateInstance(&instance_info, nullptr, &instance)){
    spdlog::error("Failed to create instance.");
    std::terminate();
  }

  vk_data.instance = instance;

  auto debug_messenger_info = VkDebugUtilsMessengerCreateInfoEXT{
    .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
    .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | 
                       VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
    .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | 
                   VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | 
                   VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
    .pfnUserCallback = debug_callback,
    .pUserData = nullptr
  };

  VkDebugUtilsMessengerEXT debug_messenger;
  
  if(CreateDebugUtilsMessengerEXT(instance, &debug_messenger_info, nullptr, &debug_messenger) != VK_SUCCESS){
    spdlog::error("Failed to set up debug messenger.");
    std::terminate();
  }

  vk_data.debug_messenger = debug_messenger;

  VkSurfaceKHR surface;

  if(glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS){
    spdlog::error("Failed to create window surface.");
    std::terminate();
  }

  vk_data.surface = surface;

  uint32_t physical_device_count = 0;
  vkEnumeratePhysicalDevices(instance, &physical_device_count, nullptr);
  VkPhysicalDevice physical_devices[physical_device_count];

  vkEnumeratePhysicalDevices(instance, &physical_device_count, physical_devices);

  auto physical_device = physical_devices[0];


  struct {
    int graphics_index = -1;
    int present_index = -1;
  } temp_indices; 

  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
  VkQueueFamilyProperties queue_families[queue_family_count];

  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families);

  int i = 0;
  VkBool32 present_support = false;
  for(auto const& queue_family : queue_families){
    if(queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT){
      temp_indices.graphics_index = i;
    }
    vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, i, surface, &present_support);
    if(present_support){
      temp_indices.present_index = i;
    }

    if(temp_indices.graphics_index != -1 && temp_indices.present_index != -1){
      break;
    }

    ++i;
  }

  auto queue_indices = Queue_Indices{
    .graphics = (uint32_t)temp_indices.graphics_index,
    .present = (uint32_t)temp_indices.present_index
  };

  auto queue_priority = 1.0f;
  auto queue_infos = std::vector<VkDeviceQueueCreateInfo>();
    queue_infos.push_back(VkDeviceQueueCreateInfo{
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = queue_indices.graphics,
      .queueCount = 1,
      .pQueuePriorities = &queue_priority
    } );
  if(queue_indices.graphics != queue_indices.present) {
    
    queue_infos.push_back( VkDeviceQueueCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = queue_indices.present,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority
      } );
  }

  auto device_features = VkPhysicalDeviceFeatures{};
  const char* device_extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  auto device_info = VkDeviceCreateInfo{
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .queueCreateInfoCount = (uint32_t)queue_infos.size(),
    .pQueueCreateInfos = queue_infos.data(),
    .enabledLayerCount = 1,
    .ppEnabledLayerNames = validation_layers,
    .enabledExtensionCount = 1,
    .ppEnabledExtensionNames = device_extensions,
    .pEnabledFeatures = &device_features
  };

  VkDevice device;
  vkCreateDevice(physical_device, &device_info, nullptr, &device);

  vk_data.device = device;

  VkQueue present_queue;
  vkGetDeviceQueue(device, queue_indices.present, 0, &present_queue);

  vk_data.present_queue = present_queue;



  VkSurfaceCapabilitiesKHR swapchain_capabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &swapchain_capabilities);

  uint32_t format_count;
  vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, nullptr);
  VkSurfaceFormatKHR formats[format_count];
  if(format_count != 0){
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, formats);
  }

  uint32_t present_mode_count;
  vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_mode_count, nullptr);
  VkPresentModeKHR present_modes[present_mode_count];
  if(present_mode_count != 0){
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_mode_count, present_modes);
  }

  VkSurfaceFormatKHR format = formats[0];
  for(auto const& available_format : formats){
    if(available_format.format == VK_FORMAT_B8G8R8A8_SRGB && available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR){
      format = available_format;
      break;
    }
  }

  VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
  for(auto const& available_present_mode : present_modes){
    if(available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR){
      present_mode = available_present_mode;
    }
  }

  VkExtent2D extent;
  if(swapchain_capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()){
    extent = swapchain_capabilities.currentExtent;
  } else {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    extent = {
      static_cast<uint32_t>(width),
      static_cast<uint32_t>(height)
    };
    
    extent.width = std::clamp(extent.width, swapchain_capabilities.minImageExtent.width, swapchain_capabilities.maxImageExtent.width);
    extent.height = std::clamp(extent.height, swapchain_capabilities.minImageExtent.height, swapchain_capabilities.maxImageExtent.height);
  }

  auto image_count = swapchain_capabilities.minImageCount + 1;

  if(swapchain_capabilities.maxImageCount > 0 && image_count > swapchain_capabilities.maxImageCount){
    image_count = swapchain_capabilities.maxImageCount;
  }

  VkSharingMode sharing_mode;
  uint32_t queue_family_index_count;
  auto queue_family_indices = std::vector<uint32_t>();

  if(queue_indices.graphics != queue_indices.present){
    sharing_mode = VK_SHARING_MODE_CONCURRENT;
    queue_family_index_count = 2;
    queue_family_indices.push_back(queue_indices.graphics);
    queue_family_indices.push_back(queue_indices.present);
  } else {
    sharing_mode = VK_SHARING_MODE_EXCLUSIVE,
    queue_family_index_count = 0;

  }

  auto swapchain_info = VkSwapchainCreateInfoKHR{
    .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
    .surface = surface,
    .minImageCount = image_count,
    .imageFormat = format.format,
    .imageColorSpace = format.colorSpace,
    .imageExtent = extent,
    .imageArrayLayers = 1,
    .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    .imageSharingMode = sharing_mode,
    .queueFamilyIndexCount = queue_family_index_count,
    .pQueueFamilyIndices = queue_family_indices.data(),
    .preTransform = swapchain_capabilities.currentTransform,
    .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    .presentMode = present_mode,
    .clipped = VK_TRUE,
    .oldSwapchain = VK_NULL_HANDLE
  };

  VkSwapchainKHR swapchain;
  if(vkCreateSwapchainKHR(device, &swapchain_info, nullptr, &swapchain) != VK_SUCCESS){
    spdlog::error("Failed to create swapchain.");
    std::terminate();
  }

  vk_data.swapchain = swapchain;
  vk_data.format = format.format;
  vk_data.extent = extent;

  uint32_t swapchain_image_count;
  vkGetSwapchainImagesKHR(device, swapchain, &swapchain_image_count, nullptr);
  auto swapchain_images = std::vector<VkImage>(swapchain_image_count);
  
  vkGetSwapchainImagesKHR(device, swapchain, &swapchain_image_count, swapchain_images.data());

  
  auto swapchain_image_views = std::vector<VkImageView>(swapchain_image_count);
  for( int i = 0; i < swapchain_images.size(); i++){
    auto image_info = VkImageViewCreateInfo{
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = swapchain_images.at(i),
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = format.format,
      .components = {
        .r = VK_COMPONENT_SWIZZLE_IDENTITY,
        .g = VK_COMPONENT_SWIZZLE_IDENTITY,
        .b = VK_COMPONENT_SWIZZLE_IDENTITY,
        .a = VK_COMPONENT_SWIZZLE_IDENTITY
      },
      .subresourceRange = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
      }
    };
    if(vkCreateImageView(device, &image_info, nullptr, &swapchain_image_views.at(i)) != VK_SUCCESS){
      spdlog::error("Failed to create image views.");
      std::terminate();
    }

    vk_data.swapchain_image_views = swapchain_image_views;
  }

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

    auto const shader_info = VkShaderModuleCreateInfo{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = buffer.size(),
      .pCode = reinterpret_cast<uint32_t const *>(buffer.data())
    };

    // return device->createShaderModuleUnique(shader_info);
    VkShaderModule shader_module;
    if(vkCreateShaderModule(device, &shader_info, nullptr, &shader_module) != VK_SUCCESS){
      spdlog::error("Failed to create shader module.");
      std::terminate();
    }

    return shader_module;
  };

  spdlog::info("Loading shader modules");
  auto vert_shader = load_shader_module("./vert.spv");
  spdlog::info("Loaded vert shader");
  auto frag_shader = load_shader_module("./frag.spv");
  spdlog::info("Loaded frag shader");


  auto vert_shader_stage_info = VkPipelineShaderStageCreateInfo{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    .stage = VK_SHADER_STAGE_VERTEX_BIT,
    .module = vert_shader,
    .pName = "main"
  };

  auto frag_shader_stage_info = VkPipelineShaderStageCreateInfo{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
    .module = frag_shader,
    .pName = "main"
  };

  VkPipelineShaderStageCreateInfo shader_stages[] = {vert_shader_stage_info, frag_shader_stage_info};

  auto vertex_input_info = VkPipelineVertexInputStateCreateInfo{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    .vertexBindingDescriptionCount = 0,
    .pVertexBindingDescriptions = nullptr,
    .vertexAttributeDescriptionCount = 0,
    .pVertexAttributeDescriptions = nullptr
  };

  auto input_assembly = VkPipelineInputAssemblyStateCreateInfo{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
    .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    .primitiveRestartEnable = VK_FALSE
  };

  auto viewport = VkViewport{
    .x = 0.0f,
    .y = 0.0f,
    .width = (float) extent.width,
    .height = (float) extent.height,
    .minDepth = 0.0f,
    .maxDepth = 1.0f
  };

  auto scissor = VkRect2D{
    .offset = {0, 0},
    .extent = extent
  };
  
  std::vector<VkDynamicState> dynamic_states = {
    VK_DYNAMIC_STATE_VIEWPORT,
    VK_DYNAMIC_STATE_SCISSOR
  };

  auto dynamic_state = VkPipelineDynamicStateCreateInfo{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
    .dynamicStateCount = (uint32_t) dynamic_states.size(),
    .pDynamicStates = dynamic_states.data()
  };

  auto viewport_state = VkPipelineViewportStateCreateInfo{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
    .viewportCount = 1,
    .pViewports = &viewport,
    .scissorCount = 1,
    .pScissors = &scissor
  };

  auto rasterizer = VkPipelineRasterizationStateCreateInfo{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
    .depthClampEnable = VK_FALSE,
    .rasterizerDiscardEnable = VK_FALSE,
    .polygonMode = VK_POLYGON_MODE_FILL,
    .cullMode = VK_CULL_MODE_BACK_BIT,
    .frontFace = VK_FRONT_FACE_CLOCKWISE,
    .depthBiasEnable = VK_FALSE,
    .depthBiasConstantFactor = 0.0f,
    .depthBiasClamp = 0.0f,
    .depthBiasSlopeFactor = 0.0f,
    .lineWidth = 1.0f,
  };

  auto multisampling = VkPipelineMultisampleStateCreateInfo{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
    .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    .sampleShadingEnable = VK_FALSE,
    .minSampleShading = 1.0f,
    .pSampleMask = nullptr,
    .alphaToCoverageEnable = VK_FALSE,
    .alphaToOneEnable = VK_FALSE
  };

  // depth stencil goes here

  auto color_blend_attachment = VkPipelineColorBlendAttachmentState{
    .blendEnable = VK_FALSE,
    .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
    .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
    .colorBlendOp = VK_BLEND_OP_ADD,
    .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
    .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
    .alphaBlendOp = VK_BLEND_OP_ADD,
    .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | 
      VK_COLOR_COMPONENT_G_BIT | 
      VK_COLOR_COMPONENT_B_BIT | 
      VK_COLOR_COMPONENT_A_BIT,
  };

  // float blend_constants[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  auto color_blending = VkPipelineColorBlendStateCreateInfo{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
    .logicOpEnable = VK_FALSE,
    .logicOp = VK_LOGIC_OP_COPY,
    .attachmentCount = 1,
    .pAttachments = &color_blend_attachment,
    .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f}
  };

  auto pipeline_layout_info = VkPipelineLayoutCreateInfo{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount = 0,
    .pSetLayouts = nullptr,
    .pushConstantRangeCount = 0,
    .pPushConstantRanges = nullptr
  };

  VkPipelineLayout pipeline_layout;
  if(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS){
    spdlog::error("Failed to create pipeline layout.");
    std::terminate();
  }

  vk_data.pipeline_layout = pipeline_layout;
  
  auto color_attachment = VkAttachmentDescription{
    .format = format.format,
    .samples = VK_SAMPLE_COUNT_1_BIT,
    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
    .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR  
  };

  auto color_attachment_ref = VkAttachmentReference{
    .attachment = 0,
    .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
  };

  auto subpass = VkSubpassDescription{
    .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
    .colorAttachmentCount = 1,
    .pColorAttachments = &color_attachment_ref
  };

  auto render_pass_info = VkRenderPassCreateInfo{
    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
    .attachmentCount = 1,
    .pAttachments = &color_attachment,
    .subpassCount = 1,
    .pSubpasses = &subpass
  };

  VkRenderPass render_pass;
  if(vkCreateRenderPass(device, &render_pass_info, nullptr, &render_pass) != VK_SUCCESS){
    spdlog::error("Failed to create render pass");
    std::terminate();
  }

  vk_data.render_pass = render_pass;

  auto pipeline_info = VkGraphicsPipelineCreateInfo{
    .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
    .stageCount = 2,
    .pStages = shader_stages,
    .pVertexInputState = &vertex_input_info,
    .pInputAssemblyState = &input_assembly,
    .pViewportState = &viewport_state,
    .pRasterizationState = &rasterizer,
    .pMultisampleState = &multisampling,
    .pDepthStencilState = nullptr,
    .pColorBlendState = &color_blending,
    .pDynamicState = &dynamic_state,
    .layout = pipeline_layout,
    .renderPass = render_pass,
    .subpass = 0,
    .basePipelineHandle = VK_NULL_HANDLE,
    .basePipelineIndex = -1
  };

  VkPipeline graphics_pipeline;

  if(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &graphics_pipeline) != VK_SUCCESS){
    spdlog::error("Failed to create graphics pipeline.");
    std::terminate();
  }

  vk_data.graphics_pipeline = graphics_pipeline;

  auto swapchain_framebuffers = std::vector<VkFramebuffer>(swapchain_image_views.size());


  for(size_t i = 0; i < swapchain_image_views.size(); i++){
    VkImageView attachments[] = {
      swapchain_image_views.at(i)
    };
    auto framebuffer_info = VkFramebufferCreateInfo{
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = render_pass,
      .attachmentCount = 1,
      .pAttachments = attachments,
      .width = extent.width,
      .height = extent.height,
      .layers = 1
    };

    if(vkCreateFramebuffer(device, &framebuffer_info, nullptr, &swapchain_framebuffers.at(i)) != VK_SUCCESS){
      spdlog::error("Failed to create framebuffer.");
      std::terminate();
    }
  }

  vk_data.swapchain_framebuffers = swapchain_framebuffers;

  auto command_pool_info = VkCommandPoolCreateInfo{
    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    .queueFamilyIndex = queue_indices.graphics
  };

  VkCommandPool command_pool;
  if(vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool) != VK_SUCCESS){
    spdlog::error("Failed to create command pool.");
    std::terminate();
  }

  vk_data.command_pool = command_pool;

  auto command_buffer_alloc_info = VkCommandBufferAllocateInfo{
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool = command_pool,
    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandBufferCount = 1
  };

  VkCommandBuffer command_buffer;
  if(vkAllocateCommandBuffers(device, &command_buffer_alloc_info, &command_buffer) != VK_SUCCESS){
    spdlog::error("Failed to allocate command buffers.");
    std::terminate();
  }

  vk_data.command_buffer = command_buffer; 

  auto command_buffer_begin_info = VkCommandBufferBeginInfo{
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = 0,
    .pInheritanceInfo = nullptr
  };

  VkCommandBufferBeginInfo begin_info;
  if(vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS){
    spdlog::error("Failed to begin recording command buffer.");
    std::terminate();
  }

  // auto render_pass_begin_info{
  //   .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
  //   // .renderPass = render_pass,
  //   // .framebuffer = 
  // };



  vkDestroyShaderModule(device, frag_shader, nullptr);
  vkDestroyShaderModule(device, vert_shader, nullptr);
  return vk_data;
}




// struct Depth_Image_Handles {
//   vk::UniqueImage image;
//   vk::UniqueDeviceMemory memory;
//   vk::UniqueImageView view;
// };

// struct Image_Handles{
//   vk::UniqueImage image;
//   vk::UniqueDeviceMemory memory;
// };

// struct Vertex{
//   glm::vec3 position;
//   glm::vec3 color;
//   glm::vec2 tex_coord;
// };

// //Cube
// auto const vertices = std::vector<Vertex>{
//   {{0,    0.5,    0.0},     {1,0,1},    {1, 0}},
//   {{-0.5, 0,      0.0},     {0,1,0},    {0, 0}},
//   {{0.5,  0,      0.0},     {1,0,1},    {0, 1}},
//   {{1,    1,      0.0},     {1,1,0},    {1, 1}},
 
//   {{0,    0.5,    -0.5},     {1,0,1},    {1, 0}},
//   {{-0.5, 0,      -0.5},     {0,1,0},    {0, 0}},
//   {{0.5,  0,      -0.5},     {1,0,1},    {0, 1}},
//   {{1,    1,      -0.5},     {1,1,0},    {1, 1}},
// };

// auto const indices = std::vector<uint32_t>{
//   0, 1, 2, 2, 3, 0,
//   4, 5, 6, 6, 7, 4
// };

// struct Uniform_Buffer_Object{
//   glm::mat4 model;
//   glm::mat4 view;
//   glm::mat4 proj;
// }; 

// constexpr auto validation_layer = "VK_LAYER_KHRONOS_validation";
// constexpr auto gfx_reconstruct_layer =  "VK_LAYER_LUNARG_gfxreconstruct";
// constexpr auto bla = "VK_LAYER_LUNARG_api_dump";

// constexpr auto default_viewport(vk::Extent2D swapchain_extent){
//   return vk::Viewport{
//     .x = 0.0f, .y = 0.0f, 
//     .width = static_cast<float>(swapchain_extent.width),
//     .height = static_cast<float>(swapchain_extent.height),
//     .minDepth = 0.0f,
//     .maxDepth = 1.0f,
//   };
// }

// //Creates an object to submit commands to for the current scope.
// struct Command_Scope{
//   Command_Scope(
//       vk::UniqueDevice const & device, 
//       vk::UniqueCommandPool const & commandPool, 
//       vk::Queue const & graphicsQueue): 
//     device(device),
//     graphic_queue(graphicsQueue) {

//     auto const alloc_info = vk::CommandBufferAllocateInfo{
//       .commandPool = commandPool.get(), 
//       .level = vk::CommandBufferLevel::ePrimary, 
//       .commandBufferCount = 1
//     };
//     command_buffer = std::move(device->allocateCommandBuffersUnique(alloc_info).back());

//     command_buffer->begin(vk::CommandBufferBeginInfo{ 
//         .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
//     });
//   }
//   ~Command_Scope(){
//       command_buffer->end();

//       auto const submit_info = vk::SubmitInfo{
//         .commandBufferCount = 1,
//         .pCommandBuffers = &command_buffer.get()
//       };

//       graphic_queue.submit(std::array{submit_info});
//       device->waitIdle();
//   }

//   auto & operator ->(){return command_buffer.get();}

//   vk::UniqueDevice const & device;
//   vk::Queue const & graphic_queue;
//   vk::UniqueCommandBuffer command_buffer;
// };

// class Renderer{
// public:
//   inline void draw_frame();
//   friend inline auto create_renderer(GLFWwindow * window) noexcept;

//   inline auto get_viewport() const noexcept{
//     return default_viewport(swapchain_extent);
//   }

//   struct Synchronization{
//     vk::Semaphore image_available_semaphore;
//     vk::Semaphore render_finished_semaphore;
//     vk::Fence in_flight_fence;
//   };

//   vk::UniqueInstance instance;
//   vk::DebugUtilsMessengerEXT messenger;
//   vk::PhysicalDevice physical_device;
//   Queue_Indices queue_indices; 
//   vk::UniqueDevice device;
//   vk::UniqueSurfaceKHR surface;
//   vk::UniqueSwapchainKHR swapchain;
//   std::vector<vk::UniqueImageView> swapchain_image_views;
//   vk::UniqueRenderPass render_pass;
//   vk::UniqueDescriptorSetLayout descriptor_set_layout;
//   vk::UniqueShaderModule vert_shader;
//   vk::UniqueShaderModule frag_shader;
//   vk::UniquePipelineLayout graphics_pipeline_layout;
//   vk::UniquePipeline graphics_pipeline;
//   std::vector<vk::UniqueFramebuffer> frame_buffers;
//   Depth_Image_Handles depth_image_handles;
//   std::vector<vk::UniqueFramebuffer> depth_buffers;
//   vk::UniqueCommandPool command_pool;
//   vk::UniqueDescriptorPool descriptor_pool;
//   std::vector<vk::UniqueDescriptorSet> descriptor_sets;
//   std::vector<vk::UniqueCommandBuffer> command_buffers;
//   vk::UniqueBuffer vertex_buffer;
//   vk::UniqueDeviceMemory vertex_buffer_memory;
//   vk::UniqueBuffer index_buffer;
//   vk::UniqueDeviceMemory index_buffer_memory;
//   std::vector<std::pair<vk::UniqueBuffer, vk::UniqueDeviceMemory>> uniform_buffers;
//   Image_Handles image_handles;
//   uint32_t texture_mipmap_levels;
//   vk::UniqueImageView texture_image_view;
//   vk::UniqueSampler texture_sampler;
//   vk::Extent2D swapchain_extent;
//   uint32_t max_frames_in_flight = 2;
//   std::vector<Renderer::Synchronization> per_frame_sync;
//   std::vector<vk::Fence> swapchain_images_in_flight;
//   uint32_t current_frame = 0;
// };



// [[nodiscard]]
// inline auto create_renderer(GLFWwindow * window) noexcept try{
//   spdlog::info("Creating Renderer");

//   constexpr auto appinfo = vk::ApplicationInfo{
//     .pApplicationName = "Toy",
//     .applicationVersion = 0,
//     .pEngineName = "Toy",
//     .engineVersion = 0,
//     .apiVersion = VK_API_VERSION_1_3,
//   };

//   //TODO: don't allocate here.
//   auto layers = std::vector<const char *>(0);
//   auto const extensions = std::invoke([]{
//     auto glfwExtensionCount = uint32_t(0);
//     auto const glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

//     auto extensions = std::vector<const char *>(glfwExtensions, glfwExtensions + glfwExtensionCount);


//     extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);


//     return extensions;
//   });

//   const char * requested_layers[] = {
//     gfx_reconstruct_layer, 
//     validation_layer,
//   };

//   for(auto const & layer : vk::enumerateInstanceLayerProperties()){
//     auto const layer_name = std::string_view(layer.layerName);
//     for(auto const & requested_layer : requested_layers){
//       if(layer_name == requested_layer){
//         spdlog::info("Found layer {}", requested_layer);
//         layers.push_back(requested_layer);
//       }
//     }
//   }

//   auto const instance_info = vk::InstanceCreateInfo{
//     .pApplicationInfo = &appinfo,
//     .enabledLayerCount = (uint32_t)layers.size(),
//     .ppEnabledLayerNames = layers.data(),
//     .enabledExtensionCount = (uint32_t) extensions.size(),
//     .ppEnabledExtensionNames = extensions.data(),
//   };

//   auto instance = vk::createInstanceUnique(instance_info);

//   auto const messengerInfo = vk::DebugUtilsMessengerCreateInfoEXT{
//     .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
//       | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
//       | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning,
//     .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation,
//     .pfnUserCallback = debugCallback,
//     .pUserData = nullptr
//   };

//   auto messenger = instance->createDebugUtilsMessengerEXT(messengerInfo, nullptr, vk::DispatchLoaderDynamic(instance.get(), vkGetInstanceProcAddr));

//   auto surface = std::invoke([&]{
//       VkSurfaceKHR surface;
//       if(glfwCreateWindowSurface(*instance, window, nullptr, &surface)){
//         spdlog::error("Unable to create window surface");
//         std::abort();
//       }

//       return vk::UniqueSurfaceKHR(surface, vk::ObjectDestroy<vk::Instance, vk::DispatchLoaderStatic>(instance.get()));
//   });

//   auto const physical_device = std::invoke([&]{
//       //TODO:
//       auto constexpr physical_device_has_ideal_properties = [](vk::PhysicalDevice const & device){
//         return true;
//       };

//       for(auto physical_device : instance->enumeratePhysicalDevices())
//       {
//         if(physical_device_has_ideal_properties(physical_device)){
//           return physical_device;
//         }
//       }

//       spdlog::error("No viable gpu");
//       std::abort();
//   });

//   auto const queue_indices = std::invoke([&] noexcept{ 
//       struct {
//         int graphics_index = -1;
//         int present_index = -1;
//       } temp_indices; 

//       auto const properties = physical_device.getQueueFamilyProperties();

//       for(auto i = 0; i < properties.size(); ++i){
//         if(temp_indices.graphics_index < 0 and properties[i].queueFlags & vk::QueueFlagBits::eGraphics){
//           spdlog::info("Found graphics index {}", 1);
//           temp_indices.graphics_index = i;
//         }
          
//         if(temp_indices.present_index < 0 and physical_device.getSurfaceSupportKHR(1, *surface)) {
//           //spdlog::info("Found present iindex {}", i);
//           temp_indices.present_index = i;
//         }

//         if(temp_indices.graphics_index >= 0 and temp_indices.present_index >= 0){
//           return Queue_Indices{
//             .graphics = (uint32_t)temp_indices.graphics_index,
//             .present = (uint32_t)temp_indices.present_index
//           };
//         }
//       }

//       spdlog::error("Unable to find graphics card that can display");
//       std::abort();
//   });

//   auto const graphics_family_index = queue_indices.graphics;
//   auto const present_family_index = queue_indices.present;

//   static auto graphicsQueuePriority = 1.0f;

//   auto const queueCreateInfos = std::array{
//     vk::DeviceQueueCreateInfo{
//       .queueFamilyIndex = (uint32_t)graphics_family_index,
//       .queueCount = 1,
//       .pQueuePriorities = &graphicsQueuePriority,
//     },
//   };

//   auto const device_extensions = std::array{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

//   auto device = physical_device.createDeviceUnique(
//       vk::DeviceCreateInfo{ 
//         .queueCreateInfoCount = queueCreateInfos.size(),
//         .pQueueCreateInfos = queueCreateInfos.data(),
//         .enabledLayerCount = static_cast<uint32_t>(layers.size()),
//         .ppEnabledLayerNames = layers.data(),
//         .enabledExtensionCount = device_extensions.size(),
//         .ppEnabledExtensionNames = device_extensions.data()
//       });

//   auto const graphics_queue = device->getQueue(graphics_family_index, 0);

//   auto const capabilities = physical_device.getSurfaceCapabilitiesKHR(surface.get());
//   //TODO: pcik a better format
//   auto const surface_format = physical_device.getSurfaceFormatsKHR(surface.get()).back();
//   //TODO: pick a better present mode
//   auto const present_mode = physical_device.getSurfacePresentModesKHR(surface.get()).back();

//   auto const image_count = capabilities.minImageCount + 1;

//   auto swapchain_extent = std::invoke([&]{
//       if(capabilities.currentExtent.width not_eq UINT32_MAX)
//         return capabilities.currentExtent;

//       int width, height;
//       glfwGetFramebufferSize(window, &width, &height);

//       auto const [
//         min_extent_width,
//         min_extent_height
//       ] = capabilities.minImageExtent;

//       auto const [
//         max_extent_width,
//         max_extent_height
//       ] = capabilities.maxImageExtent;

//       return vk::Extent2D{
//         .width = std::clamp(static_cast<uint32_t>(width), min_extent_width, max_extent_width),
//         .height = std::clamp(static_cast<uint32_t>(width), min_extent_height, max_extent_height),
//       };
//   }); 

//   auto swapchain = std::invoke([&]{ 
//       auto const queue_family_indices = std::invoke([&]{
//         auto indices = std::vector<uint32_t>();
//         indices.push_back(graphics_family_index);
//         if(present_family_index not_eq graphics_family_index){
//           indices.push_back(present_family_index);
//         } 
//         return indices;
//       });

//       auto sharing_mode = std::invoke([&]{
//         if(queue_family_indices.size() not_eq 1){
//           return vk::SharingMode::eConcurrent;
//         }else{
//           return vk::SharingMode::eExclusive;
//         }
//       });

//       auto info = vk::SwapchainCreateInfoKHR{
//         .surface = surface.get(),
//         .minImageCount = image_count,
//         .imageFormat = surface_format.format,
//         .imageColorSpace = surface_format.colorSpace,
//         .imageExtent = swapchain_extent,
//         .imageArrayLayers = 1,
//         .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
//         .imageSharingMode = sharing_mode,
//         .queueFamilyIndexCount = (uint32_t)queue_family_indices.size(),
//         .pQueueFamilyIndices = queue_family_indices.data(),
//         .preTransform = capabilities.currentTransform,
//         .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
//         .presentMode = present_mode,
//         .clipped = VK_TRUE,
//         .oldSwapchain = nullptr,
//       };

//       return device->createSwapchainKHRUnique(info);
//   });

//   auto const create_image_view = [](vk::Device const device, vk::Image const image, vk::Format const format, vk::ImageAspectFlags const aspect_flags, uint32_t mip_levels) noexcept{
//     return device.createImageViewUnique(vk::ImageViewCreateInfo{
//       .image = image,
//       .viewType = vk::ImageViewType::e2D,
//       .format = format,
//       .subresourceRange = vk::ImageSubresourceRange{
//         .aspectMask = aspect_flags,
//         .baseMipLevel = 0,
//         .levelCount = mip_levels,
//         .baseArrayLayer = 0,
//         .layerCount = 1,
//       }
//     });
//   };

//   auto swapchain_image_views = std::invoke([&]{
//       auto images = device->getSwapchainImagesKHR(swapchain.get());
//       auto image_views = std::vector<vk::UniqueImageView>(images.size());
//       for(auto i = 0; i < images.size(); ++i){
//         image_views[i] = create_image_view(device.get(), images[i], surface_format.format, vk::ImageAspectFlagBits::eColor, 1);
//       } 
//       return std::move(image_views);
//   });

//   auto render_pass = std::invoke([&] noexcept {
//     auto const color_attachment = vk::AttachmentDescription{
//       .format = surface_format.format,
//       .samples = vk::SampleCountFlagBits::e1,
//       .loadOp = vk::AttachmentLoadOp::eClear,
//       .storeOp = vk::AttachmentStoreOp::eStore,
//       .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
//       .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
//       .initialLayout = vk::ImageLayout::eUndefined,
//       .finalLayout = vk::ImageLayout::ePresentSrcKHR,
//     };


//     auto const color_attachment_ref = vk::AttachmentReference{
//       .attachment = 0, 
//       .layout = vk::ImageLayout::eColorAttachmentOptimal
//     };

//     //auto const depth_format = std::invoke([&]{
//     //  //TODO: figure out what formats are needed for depth
//     //  auto const formats ={
//     //    vk::Format::eD32Sfloat,
//     //    vk::Format::eD32SfloatS8Uint,
//     //    vk::Format::eD24UnormS8Uint
//     //  };

//     //  auto const feature = vk::FormatFeatureFlagBits::eDepthStencilAttachment;

//     //  for(auto const & format : formats){
//     //    auto const props = physical_device.getFormatProperties(format);
//     //    if((props.optimalTilingFeatures & feature) == feature){
//     //      return format;
//     //    }
//     //  }

//     //  spdlog::error("Unable to get depth format");
//     //  std::abort();
//     //});

//     //auto const depth_attachment = vk::AttachmentDescription{
//     //  .format = depth_format,
//     //  .samples = vk::SampleCountFlagBits::e1,
//     //  .loadOp = vk::AttachmentLoadOp::eClear,
//     //  .storeOp = vk::AttachmentStoreOp::eStore,
//     //  .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
//     //  .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
//     //  .initialLayout = vk::ImageLayout::eUndefined,
//     //  .finalLayout = vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal
//     //};

//     //auto const depth_attachment_ref = vk::AttachmentReference{
//     //  .attachment = 1,
//     //  .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
//     //};

//     auto const subpass = vk::SubpassDescription{
//       .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
//       .colorAttachmentCount = 1,
//       .pColorAttachments = & color_attachment_ref,
//       //.pDepthStencilAttachment = & depth_attachment_ref,
//     };

//     auto const subpass_dependency = vk::SubpassDependency{
//       .srcSubpass = VK_SUBPASS_EXTERNAL,
//       .dstSubpass = 0,
//       .srcStageMask = 
//         vk::PipelineStageFlagBits::eColorAttachmentOutput 
//         | vk::PipelineStageFlagBits::eEarlyFragmentTests,
//       .dstStageMask =
//         vk::PipelineStageFlagBits::eColorAttachmentOutput
//         | vk::PipelineStageFlagBits::eEarlyFragmentTests,
//       .srcAccessMask = {},
//       .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite
//         | vk::AccessFlagBits::eDepthStencilAttachmentWrite
//     };

//     auto const attachments = std::array{
//       color_attachment, 
//       //depth_attachment
//     };
//     auto const info = vk::RenderPassCreateInfo{
//       .flags = {},
//       .attachmentCount = attachments.size(),
//       .pAttachments = attachments.data(),
//       .subpassCount = 1,
//       .pSubpasses = &subpass,
//       .dependencyCount = 1,
//       .pDependencies = &subpass_dependency,
//     };

//     return device->createRenderPassUnique(info);
//   });

//   auto descriptor_set_layout = std::invoke([&]{

//       auto const uniform_buffer_binding = vk::DescriptorSetLayoutBinding{
//         .binding = 0,
//         .descriptorType = vk::DescriptorType::eUniformBuffer,
//         .descriptorCount = 1,
//         .stageFlags = vk::ShaderStageFlagBits::eVertex
//       };

//       auto const sampler_binding = vk::DescriptorSetLayoutBinding{
//         .binding = 1,
//         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
//         .descriptorCount = 1,
//         .stageFlags = vk::ShaderStageFlagBits::eFragment,
//       };

//       auto const descriptor_sets = std::array{uniform_buffer_binding, sampler_binding};

//       auto info = vk::DescriptorSetLayoutCreateInfo{
//         .bindingCount = descriptor_sets.size(),
//         .pBindings = descriptor_sets.data()
//       };
      
//       return device->createDescriptorSetLayoutUnique(info);
//   });

//   auto graphics_pipeline_layout = std::invoke([&]{

//       auto const info = vk::PipelineLayoutCreateInfo{
//         .setLayoutCount = 1,
//         .pSetLayouts = &descriptor_set_layout.get()
//       };

//       return device->createPipelineLayoutUnique(info);
//   });

//   auto load_shader_module = [&](std::filesystem::path shader){
//     auto file = std::ifstream(shader.string(), std::ios::ate | std::ios::binary);
//     auto error = std::error_code();
//     auto const file_size = std::filesystem::file_size(shader, error);
//     if(error){
//       spdlog::critical("Unable to read size of shader file. {}", error.message());
//       std::abort();
//     }
//     auto buffer = std::vector<char>(file_size);
//     file.seekg(0);
//     file.read(buffer.data(), file_size);

//     auto const shader_info = vk::ShaderModuleCreateInfo{
//       .codeSize = buffer.size(),
//       .pCode = reinterpret_cast<uint32_t const *>(buffer.data())
//     };

//     return device->createShaderModuleUnique(shader_info);
//   };

//   spdlog::info("Loading shader modules");
//   auto vert_shader = load_shader_module("./vert.spv");
//   spdlog::info("Loaded vert shader");
//   auto frag_shader = load_shader_module("./frag.spv");
//   spdlog::info("Loaded frag shader");

//   auto graphics_pipeline = std::invoke([&] {
//     spdlog::info("Createing render pipeline");

//     spdlog::trace("create vert shader stage info");
//     auto const vert_shader_stage = vk::PipelineShaderStageCreateInfo{
//       .stage = vk::ShaderStageFlagBits::eVertex,
//       .module = vert_shader.get(),
//       .pName = "main"
//     };

//     spdlog::trace("create frag shader stage info");
//     auto const frag_shader_stage  = vk::PipelineShaderStageCreateInfo{
//       .stage = vk::ShaderStageFlagBits::eFragment,
//       .module = frag_shader.get(),
//       .pName = "main"
//     };

//     auto const shader_stages = std::array{vert_shader_stage, frag_shader_stage};

//     spdlog::trace("creating vertex binding description");
//     //TODO: make this bind an structure of arrays.
//     auto const vertex_binding_description = vk::VertexInputBindingDescription{
//       .binding=0,
//       .stride = sizeof(Vertex),
//       .inputRate = vk::VertexInputRate::eVertex
//     };

//     spdlog::trace("creating vertex binding attributes");
//     auto const vertex_binding_attributes = std::array{
//       vk::VertexInputAttributeDescription{
//         .location = 0,
//         .binding = vertex_binding_description.binding,
//         .format = vk::Format::eR32G32B32Sfloat,
//         .offset = offsetof(Vertex, position),
//       },
//       vk::VertexInputAttributeDescription{
//         .location = 1,
//         .binding = vertex_binding_description.binding,
//         .format = vk::Format::eR32G32B32Sfloat,
//         .offset = offsetof(Vertex, color)
//       },
//       vk::VertexInputAttributeDescription{
//         .location = 2,
//         .binding = vertex_binding_description.binding,
//         .format = vk::Format::eR32G32Sfloat,
//         .offset = offsetof(Vertex, tex_coord)
//       }
//     };

//     spdlog::trace("creating vertex input state info");
//     auto const vertex_input_state = vk::PipelineVertexInputStateCreateInfo{
//       .vertexBindingDescriptionCount = 1,
//       .pVertexBindingDescriptions = &vertex_binding_description,
//       .vertexAttributeDescriptionCount = vertex_binding_attributes.size(),
//       .pVertexAttributeDescriptions = vertex_binding_attributes.data(),
//     };

//     spdlog::trace("creating input assembly state info");
//     auto const input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo{
//       .topology = vk::PrimitiveTopology::eTriangleList,
//       .primitiveRestartEnable = VK_FALSE
//     };

//     spdlog::trace("defining viewport");
//     auto const viewport = default_viewport(swapchain_extent);

//     auto const scissor = vk::Rect2D{
//       .offset = {0,0},
//       .extent = swapchain_extent
//     };

//     spdlog::trace("creating viewport state info");
//     auto const viewport_state = vk::PipelineViewportStateCreateInfo{
//       .viewportCount = 1,
//       .pViewports = &viewport,
//       .scissorCount = 1,
//       .pScissors = &scissor,
//     };

//     auto const rasterization_state = vk::PipelineRasterizationStateCreateInfo{
//       .depthClampEnable = VK_FALSE,
//       .rasterizerDiscardEnable = VK_FALSE,
//       .polygonMode = vk::PolygonMode::eFill,
//       .cullMode = vk::CullModeFlagBits::eBack,
//       .frontFace = vk::FrontFace::eCounterClockwise,
//       .depthBiasEnable = VK_FALSE,
//     };

//     auto const multisampling = vk::PipelineMultisampleStateCreateInfo{
//       .rasterizationSamples = vk::SampleCountFlagBits::e1,
//       .sampleShadingEnable = VK_FALSE,
//       .minSampleShading = 1.0f,
//       .pSampleMask = nullptr,
//       .alphaToCoverageEnable = VK_FALSE,
//       .alphaToOneEnable = VK_FALSE,
//     };

//     auto const color_blend_attachemnt = vk::PipelineColorBlendAttachmentState{
//       .blendEnable = VK_FALSE,
//       .srcColorBlendFactor = vk::BlendFactor::eOne,
//       .dstAlphaBlendFactor = vk::BlendFactor::eOne,
//       .colorWriteMask = vk::ColorComponentFlagBits::eR
//         | vk::ColorComponentFlagBits::eG
//         | vk::ColorComponentFlagBits::eB
//         | vk::ColorComponentFlagBits::eA
//     };

//     auto const blend_constants = std::array{0.0f, 0.0f, 0.0f, 0.0f};

//     auto const color_blending = vk::PipelineColorBlendStateCreateInfo{
//       .logicOpEnable = VK_FALSE,
//       .logicOp = vk::LogicOp::eCopy,
//       .attachmentCount = 1,
//       .pAttachments = &color_blend_attachemnt,
//       .blendConstants = blend_constants
//     };

//     auto const dynamic_states = std::array{
//       vk::DynamicState::eViewport,
//       vk::DynamicState::eScissor,
//       vk::DynamicState::eLineWidth
//     };

//     auto const dynamic_state = vk::PipelineDynamicStateCreateInfo{
//       .dynamicStateCount = dynamic_states.size(),
//       .pDynamicStates = dynamic_states.data()
//     };

//     auto const depth_stencil = vk::PipelineDepthStencilStateCreateInfo{
//       .depthTestEnable = VK_TRUE,
//       .depthWriteEnable = VK_TRUE,
//       .depthCompareOp = vk::CompareOp::eLessOrEqual,
//       .depthBoundsTestEnable = VK_FALSE,
//       .stencilTestEnable = VK_FALSE,
//       .minDepthBounds = 0.0f,
//       .maxDepthBounds = 1.0f
//     };

//     auto const info = vk::GraphicsPipelineCreateInfo{
//       .stageCount = shader_stages.size(), 
//       .pStages = shader_stages.data(),
//       .pVertexInputState = &vertex_input_state,
//       .pInputAssemblyState = &input_assembly_state,
//       .pViewportState = &viewport_state,
//       .pRasterizationState = &rasterization_state,
//       .pMultisampleState = &multisampling,
//       .pDepthStencilState = &depth_stencil,
//       .pColorBlendState = &color_blending,
//       .pDynamicState = &dynamic_state,
//       .layout = graphics_pipeline_layout.get(),
//       .renderPass = render_pass.get()
//     };

//     spdlog::trace("Creating graphics pipeline");
//     auto result = device->createGraphicsPipelineUnique({}, info);

//     if(static_cast<VkResult>(result.result) not_eq VK_SUCCESS){
//       spdlog::critical("Unable to create graphics pipeline");
//       std::abort();
//     }

//     return std::move(result.value);
//   }); 

//   spdlog::info("Setting up frame buffers");
//   auto frame_buffers = std::invoke([&]{
//     auto frame_buffers = std::vector<vk::UniqueFramebuffer>();
//     frame_buffers.reserve(swapchain_image_views.size());
//     for(auto const & image_view : swapchain_image_views){
        
//         auto const attachments = std::vector{ 
//             image_view.get(), 
//             //depthImageView.get() 
//         };

//         auto const info = vk::FramebufferCreateInfo{
//           .renderPass = render_pass.get(),
//           .attachmentCount = static_cast<uint32_t>(attachments.size()),
//           .pAttachments = attachments.data(),
//           .width = swapchain_extent.width,
//           .height = swapchain_extent.height,
//           .layers = 1
//         };

//         frame_buffers.push_back(device->createFramebufferUnique(info));
//     }
//     return frame_buffers;
//   });

//   auto command_pool = device->createCommandPoolUnique(vk::CommandPoolCreateInfo{
//       .queueFamilyIndex = (uint32_t)graphics_family_index
//   });

//   //spdlog::info("Setting up descriptor pools");
//   //auto descriptor_pool = std::invoke([&]{
//   //  auto const uniform_buffer_pool_size = vk::DescriptorPoolSize{
//   //    .type = vk::DescriptorType::eUniformBuffer,
//   //    .descriptorCount = (uint32_t)swapchain_image_views.size()
//   //  };

//   //  auto const texture_sampler_pool_size = vk::DescriptorPoolSize{
//   //    .type = vk::DescriptorType::eSampler,
//   //    .descriptorCount = (uint32_t)swapchain_image_views.size()
//   //  };

//   //  auto const pool_sizes = std::array{uniform_buffer_pool_size, texture_sampler_pool_size};

//   //  auto const pool_info = vk::DescriptorPoolCreateInfo{
//   //    .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
//   //    .poolSizeCount = pool_sizes.size(),
//   //    .pPoolSizes = pool_sizes.data()
//   //  };

//   //  return device->createDescriptorPoolUnique(pool_info);
//   //});

//   auto const find_memory_type_index = [&](vk::MemoryRequirements requirements, vk::MemoryPropertyFlags memory_flags)->uint32_t{
//       auto const memory_properties = physical_device.getMemoryProperties();
//     for(auto memory_index = 0; memory_index < memory_properties.memoryTypeCount; ++memory_index){
//       if(requirements.memoryTypeBits & (1 << memory_index) && (memory_properties.memoryTypes[memory_index].propertyFlags & memory_flags )){
//         return memory_index;
//       }
//     }
//     throw std::runtime_error("Unable to find memory type");
//   };

//   auto const create_buffer = [&](
//       vk::DeviceSize size,
//       vk::BufferUsageFlags usage, 
//       vk::MemoryPropertyFlags memory_properties)
//   {
//     auto buffer_info = vk::BufferCreateInfo{
//       .size = size,
//       .usage = usage,
//       .sharingMode = vk::SharingMode::eExclusive
//     };
//     auto buffer = device->createBufferUnique(buffer_info);
//     auto const buffer_memory_requirements = device->getBufferMemoryRequirements(buffer.get());
//     auto buffer_memory_info = vk::MemoryAllocateInfo{
//       .allocationSize = buffer_memory_requirements.size,
//       .memoryTypeIndex = find_memory_type_index(buffer_memory_requirements, memory_properties), 
//     };
//     auto buffer_memory = device->allocateMemoryUnique(buffer_memory_info);
//     device->bindBufferMemory(buffer.get(), buffer_memory.get(), 0);

//     struct{
//       vk::UniqueBuffer buffer;
//       vk::UniqueDeviceMemory buffer_memory;
//     } buffer_handles;
//     buffer_handles.buffer = std::move(buffer);
//     buffer_handles.buffer_memory = std::move(buffer_memory);
//     return std::move(buffer_handles);
//   };

//   auto const copy_buffer = [&](vk::UniqueBuffer & src_buffer, vk::UniqueBuffer & dst_buffer, vk::DeviceSize size){
//     auto command_scope = Command_Scope(device, command_pool, graphics_queue);
//     command_scope.command_buffer->copyBuffer(
//         *src_buffer, 
//         *dst_buffer, 
//         {vk::BufferCopy{.size = size }});
//   };
  
//   //auto uniform_buffers = std::invoke([&]{
//   //  struct uniform_buffer_handles{
//   //    vk::UniqueBuffer buffer;
//   //    vk::UniqueDeviceMemory memory;
//   //  };

//   //  auto uniformBuffers = std::vector<uniform_buffer_handles>(); 
//   //  uniformBuffers.reserve(swapchain_image_views.size());

//   //  for(auto i = 0; i < swapchain_image_views.size(); i++){
//   //      auto const buffer_size = vk::DeviceSize(sizeof(Uniform_Buffer_Object));

//   //      auto [
//   //        buffer,
//   //        memory
//   //      ] = create_buffer(buffer_size, 
//   //        vk::BufferUsageFlagBits::eUniformBuffer, 
//   //        vk::MemoryPropertyFlagBits::eHostVisible 
//   //        | vk::MemoryPropertyFlagBits::eHostCoherent
//   //      );
//   //      
//   //      uniformBuffers.push_back({std::move(buffer), std::move(memory)});
//   //  }

//   //  return uniformBuffers;
//   //});


//   //auto descriptor_sets = std::invoke([&]{

//   //  auto const layouts = std::vector(swapchain_image_views.size(), descriptor_set_layout);

//   //  auto info = vk::DescriptorSetAllocateInfo{
//   //    .descriptorPool = descriptor_pool.get(), 
//   //    .descriptorSetCount =1, 
//   //    .pSetLayouts = &descriptor_set_layout.get()
//   //  };
//   //  
//   //  auto descriptorSets = device->allocateDescriptorSetsUnique(info);

//   //  for(size_t i =0; i < swapchain_image_views.size(); i++){

//   //      auto const uniform_buffer = uniform_buffers[i].buffer.get();
//   //      auto const bufferInfo = vk::DescriptorBufferInfo {
//   //        .buffer = uniform_buffer,
//   //        .offset = 0,
//   //        .range = sizeof(Uniform_Buffer_Object)
//   //      };

//   //      auto const uniform_write_decriptor_set = vk::WriteDescriptorSet{
//   //        .descriptorCount = 1,
//   //        .
//   //      };

//   //      auto const uniformWriteDescriptorSet = vk::WriteDescriptorSet(
//   //              descriptorSets[i].get(), 
//   //              0, 
//   //              0, 
//   //              vk::DescriptorType::eUniformBuffer, 
//   //              {}, bufferInfo, {});

//   //      auto const imageInfo = vk::DescriptorImageInfo(
//   //              textureSampler, 
//   //              textureImageView, 
//   //              vk::ImageLayout::eShaderReadOnlyOptimal);

//   //      auto const samplerWriteDescriptorSet = vk::WriteDescriptorSet(
//   //              descriptorSets[i].get(), 
//   //              1, 
//   //              0, 
//   //              vk::DescriptorType::eCombinedImageSampler, 
//   //              imageInfo, {}, {}); 

//   //      device.updateDescriptorSets({
//   //                  uniformWriteDescriptorSet,
//   //                  samplerWriteDescriptorSet
//   //              }, {});
//   //  }

//   //  return std::move(descriptorSets);
//   //});



//   spdlog::info("Setting up vertex buffers");
//   auto vertex_buffer_handles = std::invoke([&]{

//     auto const buffer_size = vk::DeviceSize(sizeof(Vertex) * vertices.size());

//     auto host_buffer_handles = create_buffer(
//       buffer_size, 
//       vk::BufferUsageFlagBits::eTransferSrc, 
//       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
//     );

//     auto memory = device->mapMemory(host_buffer_handles.buffer_memory.get(), 0, buffer_size, {});
//     std::memcpy(memory, vertices.data(), buffer_size);
//     device->unmapMemory(host_buffer_handles.buffer_memory.get());

//     auto vertex_buffer_handles = create_buffer(
//       buffer_size,
//       vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, 
//       vk::MemoryPropertyFlagBits::eDeviceLocal
//     );

//     copy_buffer(host_buffer_handles.buffer, vertex_buffer_handles.buffer, buffer_size);

//     return vertex_buffer_handles;
//   });

//   spdlog::info("Settung up index buffers");
//   auto index_buffer_handles = std::invoke([&]{
//     auto const buffer_size = vk::DeviceSize(sizeof(Vertex) * vertices.size());

//     auto host_memory_flag_bits = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

//     auto host_buffer_handles = create_buffer(
//         buffer_size, 
//         vk::BufferUsageFlagBits::eTransferSrc, 
//         host_memory_flag_bits
//     );

//     auto memory = device->mapMemory(host_buffer_handles.buffer_memory.get(), 0, buffer_size, {});
//     std::memcpy(memory, vertices.data(), indices.size());
//     device->unmapMemory(host_buffer_handles.buffer_memory.get());

//     auto vertex_buffer_handles = create_buffer(
//         buffer_size,
//         vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, 
//         vk::MemoryPropertyFlagBits::eDeviceLocal);

//     copy_buffer(host_buffer_handles.buffer, vertex_buffer_handles.buffer, buffer_size);

//     return vertex_buffer_handles;
//   });


//   auto const command_buffer_info = vk::CommandBufferAllocateInfo{
//     .commandPool = command_pool.get(),
//     .level = vk::CommandBufferLevel::ePrimary,
//     .commandBufferCount =(uint32_t)frame_buffers.size() 
//   };
//   auto command_buffers = device->allocateCommandBuffersUnique(command_buffer_info);

//   spdlog::info("Creating command buffers per fram");
//   for(auto i = 0; i < frame_buffers.size(); ++i){
//     auto const & command_buffer = command_buffers[i];
//     auto const & frame_buffer = frame_buffers[i];

//     device->waitIdle();

//     command_buffer->begin(vk::CommandBufferBeginInfo{});

//     auto const clearColor = std::vector{
//         vk::ClearValue {
//           vk::ClearColorValue {
//             std::array{0.0f, 0.0f, 0.0f ,0.0f}
//           }
//         },
// //        vk::ClearValue{.depthStencil = {1.0f, 0}}
//     };

//     auto const renderArea = vk::Rect2D{
//       {0,0}, 
//       swapchain_extent
//     };

//     auto const render_pass_info = vk::RenderPassBeginInfo{
//       .renderPass = render_pass.get(), 
//       .framebuffer = frame_buffer.get(), 
//       .renderArea = renderArea, 
//       .clearValueCount = (uint32_t)clearColor.size(),
//       .pClearValues = clearColor.data(),
//     };

//     spdlog::trace("adding render pass to command buffer {}", i);
//     command_buffer->beginRenderPass(render_pass_info, vk::SubpassContents::eInline);

//     spdlog::trace("setting viewport for command buffer {}", i);
//     command_buffer->setViewport(0, default_viewport(swapchain_extent));

//     spdlog::trace("setting scissor for command buffer {}", i);
//     auto const scissor = vk::Rect2D{{0,0}, swapchain_extent};
//     command_buffer->setScissor(0, 1, &scissor);

//     command_buffer->bindPipeline(vk::PipelineBindPoint::eGraphics, graphics_pipeline.get());

//     vk::DeviceSize offsets[] = {0};

//     spdlog::trace("binding vertex buffer to command buffer {}", i);
//     auto vertex_buffers = std::array{vertex_buffer_handles.buffer.get()};
//     command_buffer->bindVertexBuffers(0, vertex_buffers.size(), vertex_buffers.data(), offsets);

//     spdlog::trace("binding index buffer to command buffer {}", i);
//     command_buffer->bindIndexBuffer(index_buffer_handles.buffer.get(), 0, vk::IndexType::eUint32);

//     //commandBuffer->bindDescriptorSets( vk::PipelineBindPoint::eGraphics, graphics_pipeline_layout.get(), 0, 0, nullptr, 0, nullptr);

//     command_buffer->draw((uint32_t)vertices.size(),1,0,0);
//     spdlog::trace("add draw indexed to command buffer {}", i);
//     command_buffer->drawIndexed(indices.size(), 1, 0, 0, 0);

//     spdlog::trace("finnishing up command buffer {}", i);
//     command_buffer->endRenderPass();
//     command_buffer->end();
//   }

//   constexpr int max_frames_in_flight = 2;

//   auto const per_frame_sync = std::invoke([&](){
//     spdlog::info("Creating frame synchronization");
//     auto per_frame_sync = std::vector<Renderer::Synchronization>(); 
//     per_frame_sync.reserve(max_frames_in_flight);

//     for(auto i =0;i < max_frames_in_flight; ++i){
//      per_frame_sync.push_back(Renderer::Synchronization{
//         .image_available_semaphore = device->createSemaphore(vk::SemaphoreCreateInfo{}),

//         .render_finished_semaphore = device->createSemaphore(vk::SemaphoreCreateInfo{}),

//         .in_flight_fence = device->createFence(vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled}), 
//       });
//     }
//     return std::move(per_frame_sync);
//   });

//   auto const swapchain_images_in_flight = std::invoke([&]{
//     spdlog::info("Creating swapchain image in flight fences");
//     auto fences = std::vector<vk::Fence>(swapchain_image_views.size());
//     for(auto i =0; i < swapchain_image_views.size(); ++i){
//       fences[i] = device->createFence(vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
//     }
//     return fences;
//   });


//   spdlog::info("Createing Renderer object");

//   auto renderer = Renderer{
//     .instance = std::move(instance),
//     .messenger = std::move(messenger),
//     .physical_device = physical_device,
//     .queue_indices = queue_indices, 
//     .device = std::move(device),
//     .surface = std::move(surface),
//     .swapchain = std::move(swapchain),
//     .swapchain_image_views = std::move(swapchain_image_views),
//     .render_pass = std::move(render_pass),
//     .descriptor_set_layout = std::move(descriptor_set_layout),
//     .vert_shader = std::move(vert_shader),
//     .frag_shader = std::move(frag_shader),
//     .graphics_pipeline_layout = std::move(graphics_pipeline_layout),
//     .graphics_pipeline = std::move(graphics_pipeline),
//     .frame_buffers = std::move(frame_buffers),
//     //.depth_image_handles = std::move(depth_image_handles),
//     //.depth_buffers = std::move(depth_buffers),
//     .command_pool = std::move(command_pool),
//     //.descriptor_pool = std::move(descriptor_pool),
//     //.descriptor_sets = std::move(descriptor_sets),
//     .command_buffers = std::move(command_buffers),
//     .vertex_buffer = std::move(vertex_buffer_handles.buffer),
//     .vertex_buffer_memory = std::move(vertex_buffer_handles.buffer_memory),
//     .index_buffer = std::move(index_buffer_handles.buffer),
//     .index_buffer_memory = std::move(index_buffer_handles.buffer_memory),
//     //.uniform_buffers = std::move(uniform_buffers),
//     //.image_handles = std::move(image_handles),
//     //.texture_mipmap_levels = std::move(texture_mipmap_levels),
//     //.texture_image_view = std::move(texture_image_view),
//     //.texture_sampler = std::move(texture_sampler),
//     .swapchain_extent = std::move(swapchain_extent),
//     .max_frames_in_flight = max_frames_in_flight,
//     .per_frame_sync = std::move(per_frame_sync),
//     .swapchain_images_in_flight = std::move(swapchain_images_in_flight),
//   };

//   spdlog::info("returning renderer");
//   return renderer;
// } catch (std::exception & exception){
//   spdlog::critical("Exception:{}", exception.what());
//   std::abort();
// }

// void Renderer::draw_frame(){
//   auto & [
//      image_available_semaphore,
//      render_finished_semaphore,
//      in_flight_fence
//   ] = per_frame_sync[current_frame];

//   spdlog::info("starting draw frame");
//   if(device->waitForFences(1, &in_flight_fence, VK_TRUE, UINT64_MAX) not_eq vk::Result::eSuccess){
//     spdlog::warn("Unable to wait for current frame {}", current_frame);
//   }

//   if(in_flight_fence){
//     device->destroyFence(in_flight_fence);
//   }

//   in_flight_fence = device->createFence(vk::FenceCreateInfo{});

//   spdlog::trace("aquireing next image");
//   auto const next_image_result = device->acquireNextImageKHR(
//     swapchain.get(), 
//     UINT64_MAX, 
//     image_available_semaphore, 
//     in_flight_fence
//   );

//   if(next_image_result.result not_eq vk::Result::eSuccess){
//     spdlog::error("failed to present swapchain image.");
//     std::abort();
//   }

//   auto image_index = next_image_result.value;
//   spdlog::trace("got image {}", image_index);
  
//   //if(imageIndex.result == vk::Result::eErrorOutOfDateKHR or imageIndex.result == vk::Result::eSuboptimalKHR or frameResized){
//   //    renderState = createVulkanRenderState(device, gpu, surface, window, graphicsIndex, presentIndex);
//   //    frameResized = false;
//   //    return;
//   //}
//   //
//   auto image_in_flight = swapchain_images_in_flight[image_index];
  
//   spdlog::trace("waiting for image in flight fence");
//   if(device->waitForFences(1, &image_in_flight, VK_TRUE, UINT64_MAX) not_eq vk::Result::eSuccess){
//     spdlog::error("Unable to wait for image in flight fence: {}", image_index);
//   }

//   image_in_flight = in_flight_fence;
  
  
//   //auto const & buffer_handles = uniform_buffers[image_index.value];

  
//   //update_uniformBuffer(
//   //        device.get(), 
//   //        bufferHandles.first.get(), 
//   //        bufferHandles.second.get(), 
//   //        renderState->swapchainExtent);
  
  
//   vk::Semaphore wait_semaphores[] = {image_available_semaphore};
//   vk::PipelineStageFlags wait_dst_stage_masks[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
//   vk::Semaphore signal_semaphores[] = {render_finished_semaphore};
//   vk::CommandBuffer graphics_queue_command_buffers[] = {command_buffers[image_index].get()};
//   auto const submit_info = vk::SubmitInfo{
//     .waitSemaphoreCount = 1, 
//     .pWaitSemaphores = wait_semaphores,
//     .pWaitDstStageMask = wait_dst_stage_masks,
//     .commandBufferCount = 1, 
//     .pCommandBuffers = graphics_queue_command_buffers,
//     .signalSemaphoreCount = 1, 
//     .pSignalSemaphores = signal_semaphores
//   };
  
//   spdlog::trace("reset fence for current frame");

//   if(device->resetFences(1, &in_flight_fence) != vk::Result::eSuccess){
//     spdlog::warn("Unable to reset fence: {}", current_frame);
//   }
  
//   auto graphics_queue = device->getQueue(queue_indices.graphics, 0);

//   spdlog::trace("submitting to graphics queue");
//   if(graphics_queue.submit(1, &submit_info, in_flight_fence) != vk::Result::eSuccess){
//     spdlog::warn("Bad submit");
//   }
  
//   auto const present_info = vk::PresentInfoKHR{
//     .waitSemaphoreCount = 1, .pWaitSemaphores = signal_semaphores,
//     .swapchainCount = 1, .pSwapchains = &swapchain.get(),
//     .pImageIndices = &image_index
//   };

//   auto present_queue = device->getQueue(queue_indices.present, 0);
  
//   spdlog::trace("presenting on present queue");
//   if(present_queue.presentKHR(present_info) != vk::Result::eSuccess){
//     spdlog::warn("Bad present");
//   }
  
//   present_queue.waitIdle();
  
//   current_frame = (current_frame + 1) % max_frames_in_flight;
// }


