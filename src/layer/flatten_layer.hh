#pragma once

#include "layer.hh"
#include "matrix/matrix.hh"

class FlattenLayer : public Layer<Matrix<Matrix<double>>, Matrix<double>>
{
public:
    size_t output_size;

public:
    FlattenLayer(const shape& input_shape);

    Matrix<double> Forward(const Matrix<Matrix<double>>& input) override;
    Matrix<Matrix<double>>
    Backward(const Matrix<double>& outputGradient) override;
};
