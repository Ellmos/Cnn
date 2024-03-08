#pragma once

#include "layer/layer.hh"
#include "matrix/matrix.hh"

template <typename T>
class ActivationLayer : public Layer<Matrix<T>, Matrix<T>>
{
public:
    Matrix<T> input;

public:
    virtual Matrix<T> Forward(const Matrix<T>& input) = 0;
    virtual Matrix<T> Backward(const Matrix<T>& outputGradient) = 0;
};
