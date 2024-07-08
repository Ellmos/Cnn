#pragma once
#include <memory>
#include <vector>

#include "layer/layer.hh"

class Neural
{
public:
    shape input_shape;

    std::vector<std::unique_ptr<Layer>> layers;

public:
    Neural(const shape inputShape);
    shape ComputePrevOutputShape();

    void AddLayer(std::unique_ptr<Layer> layer);

    Mat Forward(const Mat& input);
    Mat Backward(const Mat& gradient);
};
