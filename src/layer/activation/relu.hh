#pragma once

#include "activation_layer.hh"

template <typename T>
class Relu : public ActivationLayer<T>
{
public:
    Relu()
    {}

    Matrix<T> Forward(const Matrix<T>& input) override
    {
        LOG_TRACE("Relu::Forward");
        this->input = input.Copy();
        return this->input.Map(
            [this](const double& i) { return i > 0 ? i : 0; });
    }

    Matrix<T> Backward(const Matrix<T>& outputGradient) override
    {
        LOG_TRACE("Relu::Backward");
        return outputGradient.Copy().Map(
            [this](const double& i) { return i > 0 ? 1 : 0; });
    }
};
