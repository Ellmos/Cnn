#pragma once

#include <cstddef>

#include "matrix/matrix.hh"

struct shape
{
    size_t rows;
    size_t cols;
    size_t depth;
};

using Mat = Matrix<Matrix<double>>;

class Layer
{
public:
    shape inputShape;

public:
    virtual ~Layer(){};
    virtual Mat Forward(const Mat& input) = 0;
    virtual Mat Backward(const Mat& outputGradient) = 0;
};
