#pragma once

#include "activation_layer.hh"

class Relu : public ActivationLayer
{
public:
    Relu()
    {}

    Mat Forward(const Mat& input) override
    {
        LOG_TRACE("Softmax::Forward");
        this->input = input.Copy();

        return input;
    }

    Mat Backward(const Mat& outputGradient) override
    {
        LOG_TRACE("Softmax::Backward");
        return outputGradient.Copy().Map(
            [](const double& i) { return i > 0 ? 1 : 0; });
    }
};

// template <typename T>
// void Normalize(const Matrix<T>& input)
// {
//     double max = *max_element(inputs.begin(), inputs.end());
//     for (size_t i = 0; i < inputs.size(); i++)
//         inputs[i] -= max;
// }
//
// double Function(vector<double> inputs, size_t index)
// {
//     Normalize(inputs);
//
//     double expSum = 0;
//     for (size_t i = 0; i < inputs.size(); i++)
//     {
//         expSum += std::exp(inputs[i]);
//     }
//
//     return exp(inputs[index]) / expSum;
// }
//
// double Derivative(vector<double> inputs, size_t index)
// {
//     Normalize(inputs);
//
//     double expSum = 0;
//     for (size_t i = 0; i < inputs.size(); i++)
//     {
//         expSum += exp(inputs[i]);
//     }
//
//     double ex = exp(inputs[index]);
//
//     return (ex * expSum - ex * ex) / (expSum * expSum);
// }
