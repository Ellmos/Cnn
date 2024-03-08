#include "flatten_layer.hh"

#include "logger/logger.hh"
#include "matrix/matrix.hh"

using namespace std;

FlattenLayer::FlattenLayer(const shape& inputShape)
{
    this->inputShape = inputShape;
    this->output_size = inputShape.depth * inputShape.rows * inputShape.cols;
}

Matrix<double> FlattenLayer::Forward(const Matrix<Matrix<double>>& input)
{
    if (input.getRows() != inputShape.depth || input.getCols() != 1
        || input(0, 0).getRows() != inputShape.rows
        || input(0, 0).getCols() != inputShape.cols)
        throw invalid_argument("FlattenLayer::Forward: input matrix does not "
                               "match the shape of the layer");

    LOG_TRACE("FlattenLayer::Forward");

    Matrix<double> res = Matrix<double>(output_size, 1, false);
    size_t i = 0;
    for (size_t mat = 0; mat < inputShape.depth; ++mat)
    {
        for (size_t row = 0; row < inputShape.rows; ++row)
        {
            for (size_t col = 0; col < inputShape.cols; ++col)
            {
                res(i, 0) = input(mat, 0)(row, col);
                ++i;
            }
        }
    }
    return res;
}

Matrix<Matrix<double>>
FlattenLayer::Backward(const Matrix<double>& outputGradient)
{
    if (outputGradient.getRows()
            != inputShape.depth * inputShape.cols * inputShape.rows
        || outputGradient.getCols() != 1)
        throw invalid_argument("FlattenLayer::Backward: outputGradient matrix "
                               "does not match the shape of the layer");
    LOG_TRACE("FlattenLayer::Backward");

    Matrix<Matrix<double>> res = Matrix<Matrix<double>>(inputShape.depth, 1);
    size_t i = 0;
    for (size_t mat = 0; mat < inputShape.depth; ++mat)
    {
        Matrix<double> tmp =
            Matrix<double>(inputShape.rows, inputShape.cols, false);
        for (size_t row = 0; row < inputShape.rows; ++row)
        {
            for (size_t col = 0; col < inputShape.cols; ++col)
            {
                tmp(row, col) = outputGradient(i, 0);
                ++i;
            }
        }
        res(mat, 0) = tmp;
    }

    return res;
}
