cmake_minimum_required(VERSION 3.21.2)

project(Cnn)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Werror -Wold-style-cast -pedantic -std=c++20)

add_compile_options(-fsanitize=address)
add_link_options(-fsanitize=address)

add_executable(cnn)
target_include_directories(cnn PRIVATE src)
add_subdirectory(src)
