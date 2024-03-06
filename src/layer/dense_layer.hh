#include "layer.hh"
#include "matrix/matrix.hh"

class DenseLayer : public Layer<Matrix<double>, Matrix<double>>
{
public:
    Matrix<double> input;

    Matrix<double> weights;
    Matrix<double> biases;

public:
    DenseLayer(size_t input_size, size_t output_size);

    Matrix<double> Forward(Matrix<double> input) override;
    Matrix<double> Backward(Matrix<double> outputGradient) override;
};
