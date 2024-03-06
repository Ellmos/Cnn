#include "flatten_layer.hh"

#include "logger/logger.hh"
#include "matrix/matrix.hh"

using namespace std;

FlattenLayer::FlattenLayer(struct shape input_shape)
{
    this->input_shape = input_shape;
    this->output_size = input_shape.depth * input_shape.rows * input_shape.cols;
}

Matrix<double> FlattenLayer::Forward(Matrix<Matrix<double>> input)
{
    if (input.getRows() != input_shape.depth || input.getCols() != 1
        || input(0, 0).getRows() != input_shape.rows
        || input(0, 0).getCols() != input_shape.cols)
        throw invalid_argument("FlattenLayer::Forward: input matrix does not "
                               "match the shape of the layer");

    LOG_TRACE("FlattenLayer::Forward");

    Matrix<double> res = Matrix<double>(output_size, 1, false);
    size_t i = 0;
    for (size_t mat = 0; mat < input_shape.depth; ++mat)
    {
        for (size_t row = 0; row < input_shape.rows; ++row)
        {
            for (size_t col = 0; col < input_shape.cols; ++col)
            {
                res(i, 0) = input(mat, 0)(row, col);
                ++i;
            }
        }
    }
    return res;
}

Matrix<Matrix<double>> FlattenLayer::Backward(Matrix<double> outputGradient)
{
    if (outputGradient.getRows()
            != input_shape.depth * input_shape.cols * input_shape.rows
        || outputGradient.getCols() != 1)
        throw invalid_argument("FlattenLayer::Backward: outputGradient matrix "
                               "does not match the shape of the layer");
    LOG_TRACE("FlattenLayer::Backward");

    Matrix<Matrix<double>> res = Matrix<Matrix<double>>(input_shape.depth, 1);
    size_t i = 0;
    for (size_t mat = 0; mat < input_shape.depth; ++mat)
    {
        Matrix<double> tmp =
            Matrix<double>(input_shape.rows, input_shape.cols, false);
        for (size_t row = 0; row < input_shape.rows; ++row)
        {
            for (size_t col = 0; col < input_shape.cols; ++col)
            {
                tmp(row, col) = outputGradient(i, 0);
                ++i;
            }
        }
        res(mat, 0) = tmp;
    }

    return res;
}
