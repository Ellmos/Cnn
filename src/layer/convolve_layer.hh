#pragma once

#include "layer.hh"

class ConvolveLayer : public Layer
{
public:
    Mat input;

    Mat kernels;
    Mat biases;

public:
    ConvolveLayer(struct shape input_shape, const size_t& kernel_nbr,
                  const size_t& kernel_size);

    Mat Forward(const Mat& input) override;
    Mat Backward(const Mat& output) override;
};
