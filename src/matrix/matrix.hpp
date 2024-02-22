#pragma once

#include <cstddef>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "logger/logger.hpp"

template <typename T>
class Matrix
{
    template <typename U>
    struct is_matrix : std::false_type
    {};

    template <typename U>
    struct is_matrix<Matrix<U>> : std::true_type
    {};

public:
    size_t rows;
    size_t cols;
    std::vector<std::vector<T>> data;

public:
    Matrix()
    {
        this->rows = 0;
        this->cols = 0;
    }

    Matrix(size_t rows, size_t cols)
    {
        LOG_INFO("Matrix::constructor with %ld rows and %ld cols", rows, cols)
        this->rows = rows;
        this->cols = cols;
        data = std::vector<std::vector<T>>(rows);
        for (size_t y = 0; y < rows; y++)
            data[y] = std::vector<T>(cols);

        _fillRandomDouble(typename std::is_same<T, double>::type());
    }

    // setter
    T &operator()(size_t row, size_t col)
    {
        LOG_INFO("Matrix::operator()")
        CheckBounds(row, col);
        return data[row][col];
    }

    // getter
    const T &operator()(size_t row, size_t col) const
    {
        LOG_INFO("Matrix::operator()")
        CheckBounds(row, col);
        return this->data[row][col];
    }

    Matrix<T> operator+(const Matrix<T> &other) const
    {
        LOG_TRACE("Matrix::operator+")
        CheckDimensionAddition(other, "operator+");

        Matrix<T> result(rows, cols);

        for (size_t row = 0; row < rows; ++row)
            for (size_t col = 0; col < cols; ++col)
                result(row, col) = data[row][col] + other(row, col);
        return result;
    }

    Matrix<T> &operator+=(const Matrix<T> &other)
    {
        LOG_TRACE("Matrix::operator+=")
        CheckDimensionAddition(other, "operator+=");

        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                data[i][j] += other(i, j);

        return *this; // Return a reference to the modified matrix
    }

    Matrix<T> operator*(const Matrix<T> &other) const
    {
        LOG_TRACE("Matrix::operator*")
        CheckDimensionMultiplication(other, "operator*");

        return _customMultiplication(other, typename is_matrix<T>::type());
    }

    Matrix<T> DotCorrelate(const Matrix<T> &other) const
    {
        LOG_TRACE("Matrix::DotCorrelate")
        CheckDimensionMultiplication(other, "DotCorrelate");

        return _customMultiplication(other, typename is_matrix<T>::type(),
                                     "Correlate");
    }

    Matrix<T> Transpose(void)
    {
        LOG_TRACE("Matrix::Transpose")
        Matrix res(cols, rows);

        for (size_t row = 0; row < rows; ++row)
            for (size_t col = 0; col < cols; ++col)
                res(col, row) = data[row][col];

        return res;
    }

    Matrix<T> Flip(void)
    {
        LOG_TRACE("Matrix::Flip")
        Matrix res(rows, cols);

        for (size_t row = 0; row < rows; ++row)
        {
            for (size_t col = 0; col < cols; ++col)
            {
                res(row, col) = data[rows - 1 - row][cols - 1 - col];
            }
        }

        return res;
    }

    Matrix<T> Correlate(const Matrix<T> &kernel) const
    {

        LOG_TRACE("Matrix::Correlate")
        if (kernel.rows > rows || kernel.cols > cols)
        {
            char msg[200];
            std::sprintf(msg,
                         "Matrix::Correlate: kernel size exceeds matrix "
                         "size\nmatrix: (%ld, %ld), kernel: (%ld, %ld)",
                         rows, cols, kernel.rows, kernel.cols);
            throw std::invalid_argument(msg);
        }

        size_t res_rows = rows - kernel.rows + 1;
        size_t res_cols = cols - kernel.cols + 1;
        Matrix res(res_rows, res_cols);

        for (size_t row = 0; row < res_rows; ++row)
        {
            for (size_t col = 0; col < res_cols; ++col)
            {
                T sum = 0;
                for (size_t kRow = 0; kRow < kernel.rows; ++kRow)
                {
                    for (size_t kCols = 0; kCols < kernel.cols; ++kCols)
                    {
                        sum +=
                            data[row + kRow][col + kCols] * kernel(kRow, kCols);
                    }
                }
                res(row, col) = sum;
            }
        }

        return res;
    }

    Matrix<T> Convolve(const Matrix<T> &kernel) const
    {
        LOG_TRACE("Matrix::Convolve")
        return Correlate(kernel.Flip());
    }

    // Not supposed to erase data
    void Reshape(size_t row, size_t col)
    {
        LOG_TRACE("Matrix::Reshape")
        this->rows = row;
        this->cols = col;
        data = std::vector<std::vector<T>>(row);
        for (size_t y = 0; y < row; y++)
            data[y] = std::vector<T>(col);
    }

    std::string ToString(void) const
    {
        LOG_TRACE("Matrix::ToString")
        return _toString(typename is_matrix<T>::type());
    }

    std::string Info(void) const
    {
        LOG_TRACE("Matrix::Info")
        char buf[100];
        std::sprintf(buf, "Matrix has %ld rows and %ld columns\n", rows, cols);
        return buf;
    }

private:
    void CheckBounds(size_t row, size_t col) const
    {
        if (row >= this->rows)
        {
            std::string msg = "Matrix::operator(): row index ("
                + std::to_string(row) + ") out of range";
            throw std::out_of_range(msg);
        }
        if (col >= this->cols)
        {
            std::string msg = "Matrix::operator(): column index ("
                + std::to_string(col) + ") out of range";
            throw std::out_of_range(msg);
        }
    }

    void CheckDimensionAddition(const Matrix<T> &other,
                                const char *function) const
    {
        if (rows != other.rows || cols != other.cols)
        {
            char msg[200];
            std::sprintf(msg,
                         "Matrix::%s: matrices have different sizes\nLHS: "
                         "(%ld, %ld), RHS: (%ld, %ld)",
                         function, rows, cols, other.rows, other.cols);
            throw std::invalid_argument(msg);
        }
    }

    void CheckDimensionMultiplication(const Matrix<T> &other,
                                      const char *function) const
    {
        if (rows != other.rows || cols != other.cols)
        {
            char msg[200];
            std::sprintf(msg,
                         "Matrix::%s: matrices sizes does not match\nLHS: "
                         "(%ld, %ld), RHS: (%ld, %ld)",
                         function, rows, cols, other.rows, other.cols);
            throw std::invalid_argument(msg);
        }
    }

    // Below this is dark magic shit for different implementation of the same
    // thing when T is either a Matrix or a scalar
    Matrix<T> _customMultiplication(const Matrix<T> &other, std::true_type,
                                    std::string op = "") const
    {
        // Multiplication of matrices using given operator

        Matrix<T> res(rows, other.cols);
        for (size_t i = 0; i < rows; i++)
        {
            for (size_t j = 0; j < other.cols; j++)
            {
                T tmp = T(data[0][0].rows, data[0][0].cols);
                for (size_t k = 0; k < cols; k++)
                {
                    if (op == "Correlate")
                        tmp += data[i][k].Correlate(other(k, j));
                    else
                        tmp += data[i][k] * other(k, j);
                }
                res(i, j) = tmp;
            }
        }

        return res;
    }

    Matrix<T> _customMultiplication(const Matrix<T> &other, std::false_type,
                                    std::string op = "") const
    {
        (void)op;
        // Multiplication of scalar using normal operation

        Matrix<T> res(rows, other.cols);
        for (size_t i = 0; i < rows; i++)
        {
            for (size_t j = 0; j < other.cols; j++)
            {
                T tmp = 0;
                for (size_t k = 0; k < cols; k++)
                {
                    tmp += data[i][k] * other(k, j);
                }
                res(i, j) = tmp;
            }
        }

        return res;
    }

    std::string _toString(std::true_type) const
    {
        std::string res;
        for (size_t row = 0; row < rows; ++row)
        {
            for (size_t col = 0; col < cols; ++col)
            {
                res += "--row : " + std::to_string(row)
                    + ", col: " + std::to_string(col) + "--\n";
                res += data[row][col].ToString();
            }
            res += "\n";
        }
        return res;
    }

    std::string _toString(std::false_type) const
    {
        std::string res;
        for (size_t row = 0; row < rows; ++row)
        {
            for (size_t col = 0; col < cols; ++col)
                res += std::to_string(data[row][col]) + " ";

            res += "\n";
        }
        return res;
    }

    void _fillRandomDouble(std::false_type)
    {}

    void _fillRandomDouble(std::true_type)
    {
        std::random_device random;
        std::mt19937 gen(random());
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                data[i][j] = distribution(gen);
            }
        }
    }
};
