cmake_minimum_required(VERSION 3.16.0)
project(ToyEngine CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_BUILD_TYPE=DEBUG)
set(CMAKE_CXX_FLAGS "-fsanitize=address")

find_package(Vulkan REQUIRED FATAL_ERROR)

add_subdirectory(spdlog)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(glfw)

add_subdirectory(glm)

add_executable(toy main.cpp)
target_link_libraries(toy glfw Vulkan::Vulkan glm::glm spdlog)
target_precompile_headers(toy PUBLIC include.hpp)

