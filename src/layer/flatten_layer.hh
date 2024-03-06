#include "layer.hh"
#include "matrix/matrix.hh"

class FlattenLayer : public Layer<Matrix<Matrix<double>>, Matrix<double>>
{
public:
    struct shape input_shape;
    size_t output_size;

public:
    FlattenLayer(struct shape input_shape);

    Matrix<double> Forward(Matrix<Matrix<double>> input) override;
    Matrix<Matrix<double>> Backward(Matrix<double> outputGradient) override;
};
