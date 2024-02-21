#include "convolve_layer.hpp"

#include <stdexcept>

ConvolveLayer::ConvolveLayer(size_t kernel_size, size_t input_depth,
                             size_t output_depth)
{
    kernels(output_depth, input_depth);
    for (size_t row = 0; row < kernels.rows; ++row)
    {
        for (size_t col = 0; col < kernels.cols; ++col)
        {
            kernels(row, col).Reshape(kernel_size, kernel_size);
        }
    }
}

void *ConvolveLayer::Forward(void *input)
{
    if (!input)
        throw std::invalid_argument(
            "ConvolveLayer::forward: input matrix is NULL");

    return NULL;
}

void *ConvolveLayer::Backward(void *gradient)
{
    if (!gradient)
        throw std::invalid_argument(
            "ConvolveLayer::backward: gradient matrix is NULL");

    return NULL;
}
