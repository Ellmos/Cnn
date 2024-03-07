#pragma once

#include "activation_layer.hh"

template <typename T>
Matrix<T> ActivationLayer<T>::Forward(const Matrix<T>& input)
{
    LOG_TRACE("Activation::Forward");
    this->input = input.Copy();
    return this->input.Map([this](const double& i) { return Function(i); });
}

template <typename T>
Matrix<T> ActivationLayer<T>::Backward(const Matrix<T>& outputGradient)
{
    LOG_TRACE("Activation::Backward");
    return outputGradient.Copy().Map(
        [this](const double& i) { return Derivative(i); });
}
