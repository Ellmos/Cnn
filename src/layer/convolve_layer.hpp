#include "layer.hpp"
#include "matrix/matrix.hpp"

class ConvolveLayer : public Layer
{
public:
    Matrix<Matrix<double>> kernels;
    Matrix<Matrix<double>> biases;
    size_t depth;

public:
    ConvolveLayer(struct shape input_shape, size_t kernel_nbr,
                             size_t kernel_size);

    void *Forward(void *input) override;
    void *Backward(void *output) override;
};
