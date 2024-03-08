#include "neural.hh"

#include "layer/convolve_layer.hh"
#include "layer/flatten_layer.hh"

Neural::Neural(const shape input_shape)
    : input_shape(input_shape)
    , nbrLayers(0)
    , flattenLayer(nullptr)
{}

shape Neural::ComputePrevOutputShape()
{
    if (nbrLayers == 0)
        return input_shape;
    return input_shape;
}

void setFlattenLayer(FlattenLayer layer)
{
    (void)layer;
}

void Neural::AddLayer(ConvolveLayer layer)
{
    (void)layer;
}

template <typename T>
void Neural::AddLayer(ActivationLayer<T> layer)
{
    (void)layer;
}

void Neural::AddLayer(DenseLayer layer)
{
    (void)layer;
}

void Neural::AddLayer(PoolingLayer layer)
{
    (void)layer;
}
