#include "layer.hh"
#include "matrix/matrix.hh"

class ConvolveLayer
    : public Layer<Matrix<Matrix<double>>, Matrix<Matrix<double>>>
{
public:
    Matrix<Matrix<double>> input;

    Matrix<Matrix<double>> kernels;
    Matrix<Matrix<double>> biases;

public:
    ConvolveLayer(struct shape input_shape, size_t kernel_nbr,
                  size_t kernel_size);

    Matrix<Matrix<double>> Forward(Matrix<Matrix<double>> input) override;
    Matrix<Matrix<double>> Backward(Matrix<Matrix<double>> output) override;
};
