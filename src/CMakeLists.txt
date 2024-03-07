add_subdirectory(layer)
add_subdirectory(neural)

target_sources(cnn PRIVATE cnn.cc)
