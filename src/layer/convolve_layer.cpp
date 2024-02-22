#include "convolve_layer.hpp"

#include <stdexcept>
#include "layer/layer.hpp"

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

    size_t biases_rows = input_shape.rows - kernel_size + 1;
    size_t biases_cols = input_shape.cols - kernel_size + 1;
    biases = Matrix<Matrix<double>>(kernel_nbr, 1);
    for (size_t row = 0; row < biases.rows; ++row)
    {
        for (size_t col = 0; col < biases.cols; ++col)
        {
            biases(row, col) = Matrix<double>(biases_rows, biases_cols);
        }
    }
}

void *ConvolveLayer::Forward(void *input)
{
    if (!input)
        throw std::invalid_argument(
            "ConvolveLayer::forward: input matrix is NULL");

    Matrix<Matrix<double>> *input_matrix = (Matrix<Matrix<double>> *)input;

    // Multiplication of matrices should be cross coreelation multiplication
    Matrix<Matrix<double>> res = biases + kernels.DotCorrelate(*input_matrix);

    void *tmp = &res;

    return tmp;
}

void *ConvolveLayer::Backward(void *gradient)
{
    if (!gradient)
        throw std::invalid_argument(
            "ConvolveLayer::backward: gradient matrix is NULL");

    return NULL;
}
