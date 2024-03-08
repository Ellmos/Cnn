#pragma once

#include <cstddef>

struct shape
{
    size_t rows;
    size_t cols;
    size_t depth;
};

class LayerContainer
{};

template <typename INPUT_TYPE, typename OUTPUT_TYPE>
class Layer : public LayerContainer
{
public:
    shape inputShape;

public:
    virtual OUTPUT_TYPE Forward(const INPUT_TYPE& input) = 0;
    virtual INPUT_TYPE Backward(const OUTPUT_TYPE& outputGradient) = 0;
};
