cmake_minimum_required(VERSION 3.16.0)
project(ToyEngine CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_BUILD_TYPE=Debug)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(toy main.cpp Renderer.cpp)
target_link_libraries(toy ${CONAN_LIBS})
target_precompile_headers(toy PUBLIC include.hpp)

