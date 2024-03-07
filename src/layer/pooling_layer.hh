#pragma once

#include "layer.hh"
#include "matrix/matrix.hh"

class PoolingLayer
    : public Layer<Matrix<Matrix<double>>, Matrix<Matrix<double>>>
{
public:
    size_t poolSize;
    size_t stride;

    Matrix<Matrix<double>> input;
    Matrix<Matrix<double>> pool;

public:
    PoolingLayer(const size_t& poolingSize, const size_t& stride);

    Matrix<Matrix<double>>
    Forward(const Matrix<Matrix<double>>& input) override;
    Matrix<Matrix<double>>
    Backward(const Matrix<Matrix<double>>& output) override;

private:
    Matrix<double> UnPool(Matrix<double> mat, Matrix<double> gradient);
};
