#include "convolve_layer.hpp"

#include "layer/layer.hpp"
#include "logger/logger.hpp"
#include "matrix/matrix.hpp"

using namespace std;

ConvolveLayer::ConvolveLayer(struct shape input_shape, size_t kernel_nbr,
                             size_t kernel_size)
{
    kernels = Matrix<Matrix<double>>(kernel_nbr, input_shape.depth);
    for (size_t row = 0; row < kernels.rows; ++row)
    {
        for (size_t col = 0; col < kernels.cols; ++col)
        {
            kernels(row, col) = Matrix<double>(kernel_size, kernel_size);
        }
    }

    size_t biasesRows = input_shape.rows - kernel_size + 1;
    size_t biasesCols = input_shape.cols - kernel_size + 1;
    biases = Matrix<Matrix<double>>(kernel_nbr, 1);
    for (size_t row = 0; row < biases.rows; ++row)
    {
        for (size_t col = 0; col < biases.cols; ++col)
        {
            biases(row, col) = Matrix<double>(biasesRows, biasesCols);
        }
    }
}

Matrix<Matrix<double>> ConvolveLayer::Forward(Matrix<Matrix<double>> input)
{
    size_t inputRows = biases(0, 0).rows + kernels(0, 0).rows - 1;
    size_t inputCols = biases(0, 0).cols + kernels(0, 0).cols - 1;

    if (input.rows != kernels.cols || input.cols != 1
        || input(0, 0).rows != inputRows || input(0, 0).cols != inputCols)
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
    if (outputGradient.rows != kernels.rows || outputGradient.cols != 1
        || tmp.rows != biases(0, 0).rows || tmp.cols != biases(0, 0).cols)
        throw invalid_argument("ConvolveLayer::backward: outputGradient matrix does not "
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
