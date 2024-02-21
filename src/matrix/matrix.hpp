#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T>
class Matrix
{
    // ChatGPT dark magic to check if T is of type matrix :)
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

    Matrix(size_t row, size_t col)
    {
        this->rows = row;
        this->cols = col;
        data = std::vector<std::vector<T>>(row);
        for (size_t y = 0; y < row; y++)
            data[y] = std::vector<T>(col);
    }

    // setter
    T &operator()(size_t row, size_t col)
    {
        if (row >= this->rows || col >= this->cols)
            throw std::out_of_range("Matrix indices out of range");
        return data[row][col];
    }

    // getter
    const T &operator()(size_t rows, size_t cols) const
    {
        if (rows >= this->rows || cols >= this->cols)
            throw std::out_of_range("Matrix indices out of range");
        return this->data[rows][cols];
    }

    Matrix<T> operator+(const Matrix<T> &other) const
    {
        if (rows != other.rows || cols != other.cols)
            throw std::invalid_argument(
                "Matrix::Add: matrices have different sizes");

        Matrix<T> result(rows, cols);
        for (size_t row = 0; row < rows; ++row)
            for (size_t col = 0; col < cols; ++col)
                result(row, col) = data[row][col] + other(row, col);
        return result;
    }

    Matrix<T>& operator+=(const Matrix<T>& other) {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix::operator+=: matrices sizes do not match");
        }
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                data[i][j] += other(i, j);
            }
        }

        return *this; // Return a reference to the modified matrix
    }

    Matrix<T> operator*(const Matrix<T> &other) const
    {
        if (cols != other.rows)
            throw std::invalid_argument(
                "Matrix::Multiply: matrices sizes does not match");

        Matrix<T> res(rows, other.cols);
        for (size_t i = 0; i < rows; i++)
        {
            for (size_t j = 0; j < other.cols; j++)
            {
                T tmp;
                if constexpr (is_matrix<T>::value)
                    tmp = T(data[0][0].rows, data[0][0].cols);
                else
                    tmp = 0;

                for (size_t k = 0; k < cols; k++)
                {
                    tmp += data[i][k] * other(k, j);
                }
                res(i, j) = tmp;
            }
        }

        return res;
    }

    Matrix<T> Transpose(void)
    {
        Matrix res(cols, rows);

        for (size_t row = 0; row < rows; ++row)
            for (size_t col = 0; col < cols; ++col)
                res(col, row) = data[row][col];

        return res;
    }

    Matrix<T> Flip(void)
    {
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



    Matrix<T> Convolve(Matrix<T> kernel) const
    {
        if (kernel.rows > rows || kernel.cols > cols)
            throw std::invalid_argument(
                "Matrix::convolve: kernel size exceeds matrix size");

        size_t resRows = rows - kernel.rows + 1;
        size_t resCols = cols - kernel.cols + 1;
        Matrix res(resRows, resCols);

        for (size_t row = 0; row < resRows; ++row)
        {
            for (size_t col = 0; col < resCols; ++col)
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


    std::string toString(void)
    {
        std::string res;
        for (size_t row = 0; row < rows; ++row)
        {
            for (size_t col = 0; col < cols; ++col)
            {
                if constexpr (is_matrix<T>::value)
                {
                    res += data[row][col].toString();
                }
                else
                {
                    res += std::to_string(data[row][col]) + " ";
                }
            }
            res += "\n";
        }
        return res;
    }
};
