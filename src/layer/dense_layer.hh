#pragma once

#include "layer.hh"

class DenseLayer : public Layer
{
public:
    Mat input;

    Mat weights = Mat(1, 1, false);
    Mat biases = Mat(1, 1, false);

public:
    DenseLayer(const size_t& input_size, const size_t& output_size);

    Mat Forward(const Mat& input) override;
    Mat Backward(const Mat& outputGradient) override;
};
