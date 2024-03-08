#pragma once

#include "layer.hh"
#include "matrix/matrix.hh"

class PoolingLayer : public Layer
{
public:
    size_t poolSize;
    size_t stride;

    Mat input;
    Mat pool;

public:
    PoolingLayer(const size_t& poolingSize, const size_t& stride);

    Mat Forward(const Mat& input) override;
    Mat Backward(const Mat& output) override;

private:
    Matrix<double> UnPool(Matrix<double> mat, Matrix<double> gradient);
};
