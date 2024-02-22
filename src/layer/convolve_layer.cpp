#include "convolve_layer.hpp"

#include "layer/layer.hpp"
#include "logger/logger.hpp"

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

Matrix<Matrix<double>> ConvolveLayer::Forward(Matrix<Matrix<double>>input)
{
    LOG_TRACE("ConvolveLayer::Forward")
    return biases + kernels.DotCorrelate(input);
}

Matrix<Matrix<double>> ConvolveLayer::Backward(Matrix<Matrix<double>> output)
{
    LOG_TRACE("ConvolveLayer::Backward")
    return output;
}
