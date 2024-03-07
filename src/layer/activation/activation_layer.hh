#pragma once

#include "layer/layer.hh"
#include "matrix/matrix.hh"

template <typename T>
class ActivationLayer : public Layer<Matrix<T>, Matrix<T>>
{
public:
    Matrix<T> input;

public:
    Matrix<T> Forward(const Matrix<T>& input) override;
    Matrix<T> Backward(const Matrix<T>& outputGradient) override;

    virtual double Function(const double& i) = 0;
    virtual double Derivative(const double& i) = 0;
};

#include "activation_layer.hxx"
