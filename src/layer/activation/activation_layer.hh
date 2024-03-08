#pragma once

#include "layer/layer.hh"

class ActivationLayer : public Layer
{
public:
    Mat input;

public:
    virtual Mat Forward(const Mat& input) = 0;
    virtual Mat Backward(const Mat& outputGradient) = 0;
};
