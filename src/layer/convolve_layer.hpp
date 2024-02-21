#include "layer.hpp"
#include "matrix/matrix.hpp"

class ConvolveLayer : public Layer
{
public:
    Matrix<Matrix<double>> kernels;
    size_t depth;

public:
    ConvolveLayer(size_t kernel_size, size_t input_depth, size_t output_depth);

    void *Forward(void *input) override;
    void *Backward(void *output) override;
};
