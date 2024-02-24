#include "pooling_layer.hpp"

#include <stdexcept>

#include "logger/logger.hpp"
#include "matrix/matrix.hpp"

using namespace std;

PoolingLayer::PoolingLayer(size_t poolSize, size_t stride)
{
    this->poolSize = poolSize;
    this->stride = stride;
}

Matrix<Matrix<double>> PoolingLayer::Forward(Matrix<Matrix<double>> input)
{
    LOG_TRACE("PoolingLayer::Forward");

    Matrix<Matrix<double>> res = Matrix<Matrix<double>>(input.rows, input.cols);
    for (size_t row = 0; row < input.rows; ++row)
        for (size_t col = 0; col < input.cols; ++col)
            res(row, col) = input(row, col).Pool(this->poolSize, this->stride);

    this->input = input;
    this->pool = res;

    return res;
}

Matrix<double> PoolingLayer::UnPool(Matrix<double> input,
                                    Matrix<double> outputGradient)
{
    if (outputGradient.rows != pool(0, 0).rows
        || outputGradient.cols != pool(0, 0).cols)
        throw invalid_argument("PollingLayer::Backward: outputGradient matrix "
                               "does not match the size of the pool");

    LOG_INFO("PoolingLayer::UnPool");

    Matrix<double> gradient = Matrix<double>(input.rows, input.cols, false);

    for (size_t oRow = 0; oRow < outputGradient.rows; ++oRow)
    {
        for (size_t oCol = 0; oCol < outputGradient.cols; ++oCol)
        {
            // cout << "-----------\n";
            double value = outputGradient(oRow, oCol);
            bool found = false;

            for (size_t pRow = 0; pRow < poolSize && !found; ++pRow)
            {
                for (size_t pCol = 0; pCol < poolSize && !found; ++pCol)
                {
                    size_t r = oRow * stride + pRow;
                    size_t c = oCol * stride + pCol;
                    // cout << "r: " << r << ", c: " << c << endl;
                    if (input(r, c) != 0)
                    {
                        gradient(r, c) = value;
                        found = true;
                    }
                }
            }
        }
    }
    return gradient;
}

Matrix<Matrix<double>>
PoolingLayer::Backward(Matrix<Matrix<double>> outputGradient)
{
    LOG_TRACE("PoolingLayer::Backward");

    Matrix<Matrix<double>> res = Matrix<Matrix<double>>(input.rows, input.cols);
    for (size_t row = 0; row < input.rows; ++row)
        for (size_t col = 0; col < input.cols; ++col)
            res(row, col) = UnPool(input(row, col), outputGradient(row, col));

    return res;
}
