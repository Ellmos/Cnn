#include "dense_layer.hpp"

#include "logger/logger.hpp"
#include "matrix/matrix.hpp"

using namespace std;

DenseLayer::DenseLayer(size_t input_size, size_t output_size)
{
    weights = Matrix<double>(output_size, input_size);
    biases = Matrix<double>(output_size, 1);
}

Matrix<double> DenseLayer::Forward(Matrix<double> input)
{
    if (input.rows != weights.cols || input.cols != 1)
        throw invalid_argument("DenseLayer::Forward: input matrix does not "
                               "match the shape of the layer");

    LOG_TRACE("DenseLayer::Forward");
    this->input = input;
    return weights * input + biases;
}

Matrix<double> DenseLayer::Backward(Matrix<double> outputGradient)
{
    if (outputGradient.rows != weights.rows || outputGradient.cols != 1)
        throw invalid_argument("DenseLayer::Forward: input matrix does not "
                               "match the shape of the layer");
    LOG_TRACE("DenseLayer::Backward");

    double learningRate = 0.5;

    Matrix<double> weightsGradient = outputGradient * input.Transpose();
    Matrix<double> inputGradient = weights.Transpose() * outputGradient;

    this->weights -= weightsGradient * learningRate;
    this->biases -= outputGradient * learningRate;

    return inputGradient;
}
