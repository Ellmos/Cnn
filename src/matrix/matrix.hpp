#pragma once

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "logger/logger.hpp"

enum MatrixOperation
{
    DOT = 0,
    CORRELATE_VALID,
    CORRELATE_FULL,
    CONVOLVE_VALID,
    CONVOLVE_FULL,
    REVERSE_CORRELATE_VALID,
    REVERSE_CORRELATE_FULL,
    REVERSE_CONVOLVE_VALID,
    REVERSE_CONVOLVE_FULL,
};

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

    Matrix(size_t rows, size_t cols, bool fillRandom = true)
    {
        LOG_INFO("Matrix::constructor with %ld rows and %ld cols", rows, cols);
        this->rows = rows;
        this->cols = cols;
        data = std::vector<std::vector<T>>(rows);
        for (size_t y = 0; y < rows; y++)
            data[y] = std::vector<T>(cols);

        if (fillRandom)
            _fillRandomDouble(typename std::is_same<T, double>::type());
    }

    // setter
    T &operator()(size_t row, size_t col)
    {
        LOG_INFO("Matrix::operator()");
        CheckBounds(row, col);
        return data[row][col];
    }

    // getter
    const T &operator()(size_t row, size_t col) const
    {
        LOG_INFO("Matrix::operator()");
        CheckBounds(row, col);
        return this->data[row][col];
    }

    Matrix<T> operator+(const Matrix<T> &other) const
    {
        LOG_TRACE("Matrix::operator+ between (%ld, %ld) and (%ld, %ld)", rows,
                  cols, other.rows, other.cols);
        CheckDimensionAddition(other, "operator+");

        Matrix<T> result(rows, cols);

        for (size_t row = 0; row < rows; ++row)
            for (size_t col = 0; col < cols; ++col)
                result(row, col) = data[row][col] + other(row, col);
        return result;
    }

    Matrix<T> &operator+=(const Matrix<T> &other)
    {
        LOG_TRACE("Matrix::operator+= between (%ld, %ld) and (%ld, %ld)", rows,
                  cols, other.rows, other.cols);
        if (rows == 0 || cols == 0)
        {
            this->rows = other.rows;
            this->cols = other.cols;
            data = std::vector<std::vector<T>>(rows);
            for (size_t y = 0; y < rows; y++)
                data[y] = std::vector<T>(cols);
        }

        CheckDimensionAddition(other, "operator+=");

        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                data[i][j] += other(i, j);

        return *this;
    }

    Matrix<T> operator*(const Matrix<T> &other) const
    {
        LOG_TRACE("Matrix::operator* between (%ld, %ld) and (%ld, %ld)", rows,
                  cols, other.rows, other.cols);
        CheckDimensionMultiplication(other, "operator*");

        return _customMultiplication(other, typename is_matrix<T>::type());
    }

    Matrix<T> CustomDotProduct(const Matrix<T> &other, MatrixOperation op) const
    {
        LOG_TRACE("Matrix::CustomDotProduct");
        CheckDimensionMultiplication(other, "CustomDotProduct");

        return _customMultiplication(other, typename is_matrix<T>::type(), op);
    }

    Matrix<T> Transpose(void) const
    {
        LOG_TRACE("Matrix::Transpose");

        Matrix res(cols, rows);
        for (size_t row = 0; row < rows; ++row)
            for (size_t col = 0; col < cols; ++col)
                res(col, row) = data[row][col];

        return res;
    }

    Matrix<T> Flip(void) const
    {
        LOG_TRACE("Matrix::Flip");

        Matrix res(rows, cols);
        for (size_t row = 0; row < rows; ++row)
            for (size_t col = 0; col < cols; ++col)
                res(row, col) = data[rows - 1 - row][cols - 1 - col];

        return res;
    }

    Matrix<T> Correlate(const Matrix<T> &kernel, std::string mode) const
    {
        LOG_TRACE("Matrix::Correlate between (%ld, %ld) and (%ld, %ld) in %s",
                  rows, cols, kernel.rows, kernel.cols, mode.data());
        if (kernel.rows > rows || kernel.cols > cols)
        {
            char msg[200];
            std::sprintf(msg,
                         "Matrix::Correlate: kernel size exceeds matrix "
                         "size\nmatrix: (%ld, %ld), kernel: (%ld, %ld)",
                         rows, cols, kernel.rows, kernel.cols);
            throw std::invalid_argument(msg);
        }
        else if (mode != "valid" && mode != "full")
        {
            throw std::invalid_argument(
                "Matrix::Correlate: invalid mode given: " + mode);
        }

        size_t modifier = mode == "full" ? 1 : -1;
        size_t resRows = rows + modifier * (kernel.rows - 1);
        size_t resCols = cols + modifier * (kernel.cols - 1);
        Matrix res(resRows, resCols);

        for (size_t resRow = 0; resRow < resRows; ++resRow)
        {
            for (size_t resCol = 0; resCol < resCols; ++resCol)
            {
                T val;
                if (mode == "valid")
                    val = _correlateValid(kernel, resRow, resCol);
                else
                    val = _correlateFull(kernel, resRow, resCol);

                res(resRow, resCol) = val;
            }
        }

        return res;
    }

private:
    T _correlateValid(const Matrix<T> &kernel, size_t resRow,
                      size_t resCol) const
    {
        T sum = 0;
        for (size_t kRow = 0; kRow < kernel.rows; ++kRow)
            for (size_t kCols = 0; kCols < kernel.cols; ++kCols)
                sum +=
                    data[resRow + kRow][resCol + kCols] * kernel(kRow, kCols);

        return sum;
    }

    T _correlateFull(const Matrix<T> &kernel, size_t resRow,
                     size_t resCol) const
    {
        T sum = 0;

        for (size_t row = 0; row < rows; ++row)
        {
            for (size_t col = 0; col < cols; ++col)
            {
                size_t r = resRow - row;
                size_t c = resCol - col;
                if (resRow >= row && resCol >= col && r < kernel.rows
                    && c < kernel.cols)
                {
                    sum += data[row][col]
                        * kernel(kernel.rows - r - 1, kernel.cols - c - 1);
                }
            }
        }

        return sum;
    }

public:
    Matrix<T> Convolve(const Matrix<T> &kernel, std::string mode) const
    {
        LOG_TRACE("Matrix::Convolve");
        return this->Correlate(kernel.Flip(), mode);
    }

    Matrix<T> Zeros(void)
    {
        Matrix<T> res = Matrix(rows, cols, false);
        for (size_t row = 0; row < rows; ++row)
            for (size_t col = 0; col < cols; ++col)
                res(row, col) = _zeros(data[row][col]);

        return res;
    }

    // Not supposed to erase data
    void Reshape(size_t row, size_t col)
    {
        LOG_TRACE("Matrix::Reshape");
        this->rows = row;
        this->cols = col;
        data = std::vector<std::vector<T>>(row);
        for (size_t y = 0; y < row; y++)
            data[y] = std::vector<T>(col);
    }

    std::string ToString(void) const
    {
        LOG_INFO("Matrix::ToString");
        return _toString(typename is_matrix<T>::type());
    }

    std::string Info(void) const
    {
        LOG_INFO("Matrix::Info");
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
        if (cols != other.rows)
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

    T _performOperation(const T &a, const T &b, MatrixOperation op) const
    {
        if (op == DOT)
            return a * b;
        else if (op == CORRELATE_VALID || op == REVERSE_CORRELATE_VALID)
            return a.Correlate(b, "valid");
        else if (op == CORRELATE_FULL || op == REVERSE_CORRELATE_FULL)
            return a.Correlate(b, "full");
        else if (op == CONVOLVE_VALID || op == REVERSE_CONVOLVE_VALID)
            return a.Convolve(b, "valid");
        else if (op == CONVOLVE_FULL || op == REVERSE_CONVOLVE_FULL)
            return a.Convolve(b, "full");
        throw std::invalid_argument("This is not possible bro");
    }
    Matrix<T> _customMultiplication(const Matrix<T> &other, std::true_type,
                                    MatrixOperation op = DOT) const
    {
        Matrix<T> res(rows, other.cols);
        for (size_t i = 0; i < rows; i++)
        {
            for (size_t j = 0; j < other.cols; j++)
            {
                T tmp = T();
                for (size_t k = 0; k < cols; k++)
                {
                    if (op < REVERSE_CORRELATE_VALID)
                        tmp += _performOperation(data[i][j], other(k, j), op);
                    else
                        tmp += _performOperation(other(k, j), data[i][k], op);
                }
                res(i, j) = tmp;
            }
        }

        return res;
    }

    Matrix<T> _customMultiplication(const Matrix<T> &other, std::false_type,
                                    MatrixOperation op = DOT) const

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
        std::random_device rand;
        std::mt19937 gen(rand());
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                data[i][j] = distribution(gen);
            }
        }
    }

    T _zeros(T val, std::true_type)
    {
        return val.Zeros();
    }

    T _zeros(T val, std::false_type)
    {
        return val;
    }
};
