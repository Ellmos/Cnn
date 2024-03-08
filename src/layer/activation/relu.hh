#pragma once

#include "activation_layer.hh"

class Relu : public ActivationLayer
{
public:
    Relu()
    {}

    Mat Forward(const Mat& input) override
    {
        LOG_TRACE("Relu::Forward");
        this->input = input.Copy();
        return this->input.Map([](const double& i) { return i > 0 ? i : 0; });
    }

    Mat Backward(const Mat& outputGradient) override
    {
        LOG_TRACE("Relu::Backward");
        return outputGradient.Copy().Map(
            [](const double& i) { return i > 0 ? 1 : 0; });
    }
};
