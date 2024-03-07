#include "pooling_layer.hh"

#include <stdexcept>

#include "logger/logger.hh"
#include "matrix/matrix.hh"

using namespace std;

PoolingLayer::PoolingLayer(const size_t& poolSize, const size_t& stride)
{
    this->poolSize = poolSize;
    this->stride = stride;
}

Matrix<Matrix<double>>
PoolingLayer::Forward(const Matrix<Matrix<double>>& input)
{
    LOG_TRACE("PoolingLayer::Forward");

    this->input = input.Copy();

    Matrix<Matrix<double>> res =
        Matrix<Matrix<double>>(input.getRows(), input.getCols());
    for (size_t row = 0; row < input.getRows(); ++row)
        for (size_t col = 0; col < input.getCols(); ++col)
            res(row, col) =
                this->input(row, col).Pool(this->poolSize, this->stride);

    this->pool = res;

    return res;
}

Matrix<double> PoolingLayer::UnPool(Matrix<double> input,
                                    Matrix<double> outputGradient)
{
    if (outputGradient.getRows() != pool(0, 0).getRows()
        || outputGradient.getCols() != pool(0, 0).getCols())
        throw invalid_argument("PollingLayer::Backward: outputGradient matrix "
                               "does not match the size of the pool");

    LOG_INFO("PoolingLayer::UnPool");

    Matrix<double> gradient =
        Matrix<double>(input.getRows(), input.getCols(), false);

    for (size_t oRow = 0; oRow < outputGradient.getRows(); ++oRow)
    {
        for (size_t oCol = 0; oCol < outputGradient.getCols(); ++oCol)
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
PoolingLayer::Backward(const Matrix<Matrix<double>>& outputGradient)
{
    LOG_TRACE("PoolingLayer::Backward");

    Matrix<Matrix<double>> res =
        Matrix<Matrix<double>>(input.getRows(), input.getCols());
    for (size_t row = 0; row < input.getRows(); ++row)
        for (size_t col = 0; col < input.getCols(); ++col)
            res(row, col) = UnPool(input(row, col), outputGradient(row, col));

    return res;
}
