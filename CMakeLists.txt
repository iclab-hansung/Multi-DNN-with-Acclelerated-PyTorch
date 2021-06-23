cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(inmo)

include_directories(~/inmo/include)
include_directories(~/.local/lib/python3.6/site-packages/torch/include/torch/csrc/api/include)
include_directories(~/.local/lib/python3.6/site-packages/torch/include)

link_directories(/usr/local/lib)
link_directories(~/.local/lib/python3.6/site-packages/torch/lib)

set(CMAKE_PREFIX_PATH ~/.local/lib/python3.6/site-packages/torch)
set(Boost_USE_MULTITHREADED ON)
set(Torch_DIR ~/.local/lib/python3.6/site-packages/torch)

find_package(Torch REQUIRED)

file(GLOB SOURCES "src/*")

add_executable(inmo ${SOURCES})
target_link_libraries(inmo "${TORCH_LIBRARIES}")
set_property(TARGET inmo PROPERTY CXX_STANDARD 14)
