#include "neural.hh"

#include <iostream>
#include <memory>
#include <utility>

#include "logger/logger.hh"

Neural::Neural(const shape input_shape)
    : input_shape(input_shape)
{}

shape Neural::ComputePrevOutputShape()
{
    if (layers.size() == 0)
        return input_shape;
    return input_shape;
}

void Neural::AddLayer(std::unique_ptr<Layer> layer)
{
    LOG_TRACE("Neural::AddLayer");
    layers.push_back(std::move(layer));
}

// Has to be called with at least one layer or will go kaboom
// Not doping a check here for optimisation
Mat Neural::Forward(const Mat& input)
{
    LOG_TRACE("Neural::Forward");

    Mat res = layers[0]->Forward(input);
    for (size_t i = 1; i < layers.size(); ++i)
    {
        res = layers[i]->Forward(res);
        //
        // std::cout << "\n-------- Forward: Layer: " << i << " ----------\n";
        // std::cout << res.Info();
        // std::cout << res(0, 0).Info();
    }
    return res;
}

// Has to be called with at least one layer or will go kaboom
// Not doping a check here for optimisation
Mat Neural::Backward(const Mat& gradient)
{
    LOG_TRACE("Neural::Backward");

    Mat res = layers[layers.size() - 1]->Backward(gradient);
    for (size_t i = layers.size() - 1; i > 0; --i)
    {
        res = layers[i - 1]->Backward(res);
        // std::cout << "\n--------Backward: Layer: " << i << " ----------\n";
        // std::cout << res.Info();
        // std::cout << res(0, 0).Info();
    }
    return res;
}
