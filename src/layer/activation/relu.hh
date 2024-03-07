#pragma once

#include "activation_layer.hh"

template <typename T>
class Relu : public ActivationLayer<T>
{
public:
    Relu();
    double Function(const double& i) override;
    double Derivative(const double& i) override;
};

#include "relu.hxx"
