#pragma once

#include <memory>
#include <vector>

#include "layer/activation/activation_layer.hh"
#include "layer/convolve_layer.hh"
#include "layer/dense_layer.hh"
#include "layer/layer.hh"
#include "layer/pooling_layer.hh"
class Neural
{
public:
    shape input_shape;
    size_t nbrLayers;

    std::vector<std::unique_ptr<LayerContainer>> convolveLayers;
    std::unique_ptr<LayerContainer> flattenLayer;
    std::vector<std::unique_ptr<LayerContainer>> denseLayers;

public:
    Neural(const shape inputShape);
    shape ComputePrevOutputShape();

    void setFlattenLayer(LayerContainer layer);

    template <typename T>
    void AddLayer(ActivationLayer<T> layer);
    void AddLayer(ConvolveLayer layer);
    void AddLayer(PoolingLayer layer);
    void AddLayer(DenseLayer layer);
};
