#pragma once

#include "layer.hh"

class FlattenLayer : public Layer
{
public:
    size_t output_size;

public:
    FlattenLayer(const shape& input_shape);

    Mat Forward(const Mat& input) override;
    Mat Backward(const Mat& outputGradient) override;
};
