#pragma once

#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

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

    Matrix(size_t row, size_t col)
    {
        this->rows = row;
        this->cols = col;
        data = std::vector<std::vector<T>>(row);
        for (size_t y = 0; y < row; y++)
            data[y] = std::vector<T>(col);

        FillRandomDouble(typename std::is_same<T, double>::type());
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

    Matrix<T> &operator+=(const Matrix<T> &other)
    {
        if (rows != other.rows || cols != other.cols)
        {
            throw std::invalid_argument(
                "Matrix::operator+=: matrices sizes do not match");
        }
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
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

        return multiply(other, typename is_matrix<T>::type());
    }

public:
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

    Matrix<T> Correlate(Matrix<T> kernel) const
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

    Matrix<T> Convolve(Matrix<T> kernel) const
    {
        return Correlate(kernel.Flip());
    }

    void Reshape(size_t row, size_t col)
    {
        this->rows = row;
        this->cols = col;
        data = std::vector<std::vector<T>>(row);
        for (size_t y = 0; y < row; y++)
            data[y] = std::vector<T>(col);
    }

    std::string ToString(void) const
    {
        return _toString(typename is_matrix<T>::type());
    }

// Below this is dark magic shit differnet implementation of the same
// focking shit to handle matrices of matrices

private:
    Matrix<T> multiply(const Matrix<T> &other, std::true_type) const
    {
        Matrix<T> res(rows, other.cols);
        for (size_t i = 0; i < rows; i++)
        {
            for (size_t j = 0; j < other.cols; j++)
            {
                T tmp = T(data[0][0].rows, data[0][0].cols);
                for (size_t k = 0; k < cols; k++)
                {
                    tmp += data[i][k] * other(k, j);
                }
                res(i, j) = tmp;
            }
        }

        return res;
    }

    Matrix<T> multiply(const Matrix<T> &other, std::false_type) const
    {
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

    void FillRandomDouble(std::false_type)
    {}

    void FillRandomDouble(std::true_type)
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
