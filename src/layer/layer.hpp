#pragma once

#include <cstddef>

struct shape
{
    size_t rows;
    size_t cols;
    size_t depth;
};

class Layer
{
public:
    virtual void *Forward(void *input) = 0;
    virtual void *Backward(void *output) = 0;
};
