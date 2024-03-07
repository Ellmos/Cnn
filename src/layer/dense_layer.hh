#pragma once

#include "layer.hh"
#include "matrix/matrix.hh"

class DenseLayer : public Layer<Matrix<double>, Matrix<double>>
{
public:
    Matrix<double> input;

    Matrix<double> weights;
    Matrix<double> biases;

public:
    DenseLayer(const size_t& input_size, const size_t& output_size);

    Matrix<double> Forward(const Matrix<double>& input) override;
    Matrix<double> Backward(const Matrix<double>& outputGradient) override;
};
