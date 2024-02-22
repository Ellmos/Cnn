#pragma once

#include <cstddef>

struct shape
{
    size_t rows;
    size_t cols;
    size_t depth;
};

template <typename INPUT_TYPE, typename OUTPUT_TYPE>
class Layer
{
public:
    virtual OUTPUT_TYPE Forward(INPUT_TYPE input) = 0;
    virtual INPUT_TYPE Backward(OUTPUT_TYPE output) = 0;
};
