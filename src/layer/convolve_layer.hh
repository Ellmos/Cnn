#pragma once

#include "layer.hh"
#include "matrix/matrix.hh"

using M = Matrix<Matrix<double>>;

class ConvolveLayer : public Layer<M, M>
{
public:
    M input;

    M kernels;
    M biases;

public:
    ConvolveLayer(struct shape input_shape, const size_t& kernel_nbr,
                  const size_t& kernel_size);

    M Forward(const M& input) override;
    M Backward(const M& output) override;
};
