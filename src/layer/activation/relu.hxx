#pragma once

#include "relu.hh"

template <typename T>
Relu<T>::Relu()
{}

template <typename T>
double Relu<T>::Function(const double& i)
{
    return i > 0 ? i : 0;
}

template <typename T>
double Relu<T>::Derivative(const double& i)
{
    return i > 0 ? 1 : 0;
}
