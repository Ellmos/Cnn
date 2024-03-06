#include "convolve_layer.hh"

#include "layer/layer.hh"
#include "logger/logger.hh"
#include "matrix/matrix.hh"

using namespace std;

ConvolveLayer::ConvolveLayer(struct shape input_shape, size_t kernel_nbr,
                             size_t kernel_size)
{
    kernels = Matrix<Matrix<double>>(kernel_nbr, input_shape.depth);
    for (size_t row = 0; row < kernels.getRows(); ++row)
    {
        for (size_t col = 0; col < kernels.getCols(); ++col)
        {
            kernels(row, col) = Matrix<double>(kernel_size, kernel_size);
        }
    }

    size_t biasesRows = input_shape.rows - kernel_size + 1;
    size_t biasesCols = input_shape.cols - kernel_size + 1;
    biases = Matrix<Matrix<double>>(kernel_nbr, 1);
    for (size_t row = 0; row < biases.getRows(); ++row)
    {
        for (size_t col = 0; col < biases.getCols(); ++col)
        {
            biases(row, col) = Matrix<double>(biasesRows, biasesCols);
        }
    }
}

Matrix<Matrix<double>> ConvolveLayer::Forward(Matrix<Matrix<double>> input)
{
    size_t inputRows = biases(0, 0).getRows() + kernels(0, 0).getRows() - 1;
    size_t inputCols = biases(0, 0).getCols() + kernels(0, 0).getCols() - 1;

    if (input.getRows() != kernels.getCols() || input.getCols() != 1
        || input(0, 0).getRows() != inputRows
        || input(0, 0).getCols() != inputCols)
        throw invalid_argument("ConvolveLayer::Forward: input matrix does not "
                               "match the shape of the layer");

    LOG_TRACE("ConvolveLayer::Forward");
    this->input = input;
    return biases + kernels.CustomDotProduct(input, REVERSE_CORRELATE_VALID);
}

Matrix<Matrix<double>>
ConvolveLayer::Backward(Matrix<Matrix<double>> outputGradient)
{
    Matrix<double> tmp = outputGradient(0, 0);
    if (outputGradient.getRows() != kernels.getRows()
        || outputGradient.getCols() != 1
        || tmp.getRows() != biases(0, 0).getRows()
        || tmp.getCols() != biases(0, 0).getCols())
        throw invalid_argument(
            "ConvolveLayer::backward: outputGradient matrix does not "
            "match the shape of the layer");

    LOG_TRACE("ConvolveLayer::Backward");

    Matrix<Matrix<double>> kernelsGradient = outputGradient.CustomDotProduct(
        this->input.Transpose(), REVERSE_CORRELATE_VALID);

    Matrix<Matrix<double>> inputGradient = kernels.Transpose().CustomDotProduct(
        outputGradient, REVERSE_CONVOLVE_FULL);

    double learningRate = 0.5;
    this->kernels -= kernelsGradient * learningRate;
    this->biases -= outputGradient * learningRate;

    return inputGradient;
}
