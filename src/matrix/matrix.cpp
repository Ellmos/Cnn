#include "matrix.hpp"

#include <cstddef>
#include <stdexcept>

using namespace std;

Matrix::Matrix(size_t height, size_t width)
{
    this->height = height;
    this->width = width;
    data = vector<vector<double>>(height);
    for (size_t y = 0; y < height; y++)
        data[y] = vector<double>(width);
}

Matrix *Matrix::Add(Matrix *other)
{
    if (!other)
        throw std::invalid_argument("Matrix::Add: other matrix is NULL");
    if (width != other->width || height != other->height)
        throw std::invalid_argument(
            "Matrix::Add: matrices have different sizes");

    Matrix *res = new Matrix(height, width);
    for (size_t y = 0; y < res->height; y++)
        for (size_t x = 0; x < res->width; x++)
            res->data[y][x] = data[y][x] + other->data[y][x];

    return res;
}

Matrix *Matrix::Multiply(Matrix *other)
{
    if (!other)
        throw std::invalid_argument("Matrix::Dot: other matrix is NULL");
    if (width != other->height)
        throw std::invalid_argument(
            "Matrix::Multiply: matrices sizes does not match");

    Matrix *res = new Matrix(height, other->width);

    for (size_t i = 0; i < height; i++)
    {
        for (size_t j = 0; j < other->width; j++)
        {
            double tmp = 0;
            for (size_t k = 0; k < width; k++)
            {
                tmp += data[i][k] * other->data[k][j];
            }
            res->data[i][j] = tmp;
        }
    }

    return res;
}
Matrix *Matrix::Transpose(void)
{
    Matrix *res = new Matrix(width, height);

    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            res->data[x][y] = data[y][x];
        }
    }

    return res;
}

Matrix *Matrix::Flip(void)
{
    Matrix *res = new Matrix(height, width);

    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            res->data[y][x] = data[height - 1 - y][width - 1 - y];
            res->data[height - 1 - y][width - 1 - x] = data[y][x];
        }
    }

    return res;
}

Matrix *Matrix::Convolve(Matrix *kernel)
{
    if (kernel->height > height || kernel->width > width)
        throw std::invalid_argument(
            "Matrix::convolve: kernel size exceeds matrix size");

    size_t resWidth = height - kernel->height + 1;
    size_t resHeight = width - kernel->width + 1;
    Matrix *res = new Matrix(resHeight, resWidth);

    for (size_t y = 0; y < resWidth; ++y)
    {
        for (size_t x = 0; x < resHeight; ++x)
        {
            double sum = 0;
            for (size_t ky = 0; ky < kernel->height; ++ky)
            {
                for (size_t kx = 0; kx < kernel->width; ++kx)
                {
                    sum += data[y + ky][x + kx] * kernel->data[ky][kx];
                }
            }
            res->data[y][x] = sum;
        }
    }

    return res;
}


std::string Matrix::toString()
{
    std::string res;
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            res += std::to_string(data[y][x]) + " ";
        }
        res += "\n";
    }
    return res;
}
