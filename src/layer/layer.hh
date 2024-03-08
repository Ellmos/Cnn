#pragma once

#include <cstddef>

#include "matrix/matrix.hh"

struct shape
{
    size_t rows;
    size_t cols;
    size_t depth;
};

class LayerContainer
{};

using Mat = Matrix<Matrix<double>>;

class Layer : public LayerContainer
{
public:
    shape inputShape;

public:
    virtual Mat Forward(const Mat& input) = 0;
    virtual Mat Backward(const Mat& outputGradient) = 0;
};
