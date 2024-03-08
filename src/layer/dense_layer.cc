#include "dense_layer.hh"

#include "logger/logger.hh"
#include "matrix/matrix.hh"

using namespace std;

DenseLayer::DenseLayer(const size_t& input_size, const size_t& output_size)
{
    weights(0, 0) = Matrix<double>(output_size, input_size);
    biases(0, 0) = Matrix<double>(output_size, 1);
}

Mat DenseLayer::Forward(const Mat& input)
{
    if (input.getRows() != weights.getCols() || input.getCols() != 1)
        throw invalid_argument("DenseLayer::Forward: input matrix does not "
                               "match the shape of the layer");

    LOG_TRACE("DenseLayer::Forward");
    Mat res = Mat(1, 1, false);
    this->input = input;
    res(0, 0) = weights(0, 0) * input(0, 0) + biases(0, 0);
    return res;
}

Mat DenseLayer::Backward(const Mat& outputGradient)
{
    if (outputGradient.getRows() != weights.getRows()
        || outputGradient.getCols() != 1)
        throw invalid_argument("DenseLayer::Forward: input matrix does not "
                               "match the shape of the layer");
    LOG_TRACE("DenseLayer::Backward");

    double learningRate = 0.5;

    Matrix<double> weightsGradient =
        outputGradient(0, 0) * input(0, 0).Transpose();

    this->weights(0, 0) -= weightsGradient * learningRate;
    this->biases(0, 0) -= outputGradient(0, 0) * learningRate;

    Mat res = Mat(1, 1, false);
    res(0, 0) = weights(0, 0).Transpose() * outputGradient(0, 0);
    return res;
}
