#include "dense_layer.hh"

#include "logger/logger.hh"
#include "matrix/matrix.hh"

using namespace std;

DenseLayer::DenseLayer(const size_t& input_size, const size_t& output_size)
{
    weights = Matrix<double>(output_size, input_size);
    biases = Matrix<double>(output_size, 1);
}

Matrix<double> DenseLayer::Forward(const Matrix<double>& input)
{
    if (input.getRows() != weights.getCols() || input.getCols() != 1)
        throw invalid_argument("DenseLayer::Forward: input matrix does not "
                               "match the shape of the layer");

    LOG_TRACE("DenseLayer::Forward");
    this->input = input;
    return weights * input + biases;
}

Matrix<double> DenseLayer::Backward(const Matrix<double>& outputGradient)
{
    if (outputGradient.getRows() != weights.getRows()
        || outputGradient.getCols() != 1)
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
