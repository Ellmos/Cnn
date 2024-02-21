#include "convolve_layer.hpp"

#include <stdexcept>

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
