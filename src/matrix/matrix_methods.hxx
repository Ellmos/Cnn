#pragma once

#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "logger/logger.hh"
#include "matrix/matrix.hh"

template <typename T>
Matrix<T> Matrix<T>::CustomDotProduct(const Matrix<T>& other,
                                      MatrixOperation op) const
{
    LOG_TRACE("Matrix::CustomDotProduct");
    _checkDimensionMultiplication(other, "CustomDotProduct");

    Matrix<T> res(rows_, other.cols_);
    for (size_t i = 0; i < rows_; ++i)
    {
        for (size_t j = 0; j < other.cols_; ++j)
        {
            T tmp = T();
            for (size_t k = 0; k < cols_; ++k)
            {
                if (op < REVERSE_CORRELATE_VALID)
                    tmp += _applyDotProduct(data_[i][j], other(k, j), op);
                else
                    tmp += _applyDotProduct(other(k, j), data_[i][k], op);
            }
            res(i, j) = tmp;
        }
    }

    return res;
}

template <typename T>
Matrix<T> Matrix<T>::Transpose() const
{
    LOG_TRACE("Matrix::Transpose");

    Matrix res(cols_, rows_);
    for (size_t row = 0; row < rows_; ++row)
        for (size_t col = 0; col < cols_; ++col)
            res(col, row) = data_[row][col];

    return res;
}

template <typename T>
Matrix<T> Matrix<T>::Flip() const
{
    LOG_TRACE("Matrix::Flip");

    Matrix res(rows_, cols_);
    for (size_t row = 0; row < rows_; ++row)
        for (size_t col = 0; col < cols_; ++col)
            res(row, col) = data_[rows_ - 1 - row][cols_ - 1 - col];

    return res;
}

template <typename T>
Matrix<T> Matrix<T>::Correlate(const Matrix<T>& kernel,
                               const std::string mode) const
{
    _checkCorrelateArgs(kernel, mode, "Correlate");

    LOG_TRACE("Matrix::Correlate between (%ld, %ld) and (%ld, %ld) in %s",
              rows_, cols_, kernel.rows_, kernel.cols_, mode.data());

    size_t modifier = mode == "full" ? 1 : -1;
    size_t resRows = rows_ + modifier * (kernel.rows_ - 1);
    size_t resCols = cols_ + modifier * (kernel.cols_ - 1);
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

template <typename T>
Matrix<T> Matrix<T>::Convolve(const Matrix<T>& kernel,
                              const std::string mode) const
{
    LOG_TRACE("Matrix::Convolve");
    return this->Correlate(kernel.Flip(), mode);
}

template <typename T>
Matrix<T> Matrix<T>::Pool(const size_t& poolSize, const size_t& stride)
{
    char msg[200];
    if (poolSize > rows_ || poolSize > cols_)
    {
        std::sprintf(msg,
                     "Matrix::Pool: pool size exceeds matrix size\n"
                     "matrix: (%ld, %ld), pool size: %ld",
                     rows_, cols_, poolSize);
        throw std::invalid_argument(msg);
    }

    LOG_TRACE("Matrix::Pool on (%ld, %ld) with a pool size of %ld", rows_,
              cols_, poolSize);

    if ((rows_ - poolSize) % stride != 0 || (cols_ - poolSize) % stride != 0)
        LOG_WARN("Matrix::Pool: Pooling is not adequat to the matrix")

    size_t resRows = (rows_ - poolSize) / stride + 1;
    size_t resCols = (cols_ - poolSize) / stride + 1;
    Matrix res(resRows, resCols);

    for (size_t resRow = 0; resRow < resRows; ++resRow)
        for (size_t resCol = 0; resCol < resCols; ++resCol)
            res(resRow, resCol) = _pool(poolSize, stride, resRow, resCol);

    return res;
}

template <typename T>
Matrix<T> Matrix<T>::Copy() const
{
    LOG_TRACE("Matrix::Copy");
    Matrix<T> res = Matrix(rows_, cols_, false);

    res.setRows(rows_);
    res.setCols(cols_);
    for (size_t row = 0; row < rows_; ++row)
    {
        for (size_t col = 0; col < cols_; ++col)
        {
            if constexpr (is_matrix<T>::value)
                res(row, col) = data_[row][col].Copy();
            else
                res(row, col) = data_[row][col];
        }
    }
    return res;
}

template <typename T>
Matrix<T> Matrix<T>::Zeros()
{
    LOG_TRACE("Matrix::Zeros");
    Matrix<T> res = Matrix(rows_, cols_, false);
    for (size_t row = 0; row < rows_; ++row)
    {
        for (size_t col = 0; col < cols_; ++col)
        {
            if constexpr (is_matrix<T>::value)
                res(row, col) = Zeros(data_[row][col]);
        }
    }

    return res;
}

template <typename T>
template <typename FUN>
Matrix<T> Matrix<T>::Map(FUN fun)
{
    LOG_TRACE("Matrix::Map");
    Matrix<T> res = Matrix(rows_, cols_, false);
    for (size_t row = 0; row < rows_; ++row)
    {
        for (size_t col = 0; col < cols_; ++col)
        {
            if constexpr (is_matrix<T>::value)
                res(row, col) = data_[row][col].Map(fun);
            else
                res(row, col) = fun(data_[row][col]);
        }
    }
    return res;
}

template <typename T>
std::string Matrix<T>::ToString() const
{
    LOG_INFO("Matrix::ToString");
    std::string res;
    for (size_t row = 0; row < rows_; ++row)
    {
        for (size_t col = 0; col < cols_; ++col)
        {
            if constexpr (is_matrix<T>::value)
            {
                res += "--row : " + std::to_string(row)
                    + ", col: " + std::to_string(col) + "--\n";
                res += data_[row][col].ToString();
            }
            else
                res += std::to_string(data_[row][col]) + " ";
        }
        res += "\n";
    }
    return res;
}

template <typename T>
std::string Matrix<T>::Info() const
{
    LOG_INFO("Matrix::Info");
    char buf[100];
    std::sprintf(buf, "Matrix has %ld rows and %ld columns\n", rows_, cols_);
    return buf;
}

template <typename T>
T Matrix<T>::_correlateValid(const Matrix<T>& kernel, const size_t& resRow,
                             const size_t& resCol) const
{
    T sum = 0;
    for (size_t kRow = 0; kRow < kernel.rows_; ++kRow)
        for (size_t kCols = 0; kCols < kernel.cols_; ++kCols)
            sum += data_[resRow + kRow][resCol + kCols] * kernel(kRow, kCols);

    return sum;
}

template <typename T>
T Matrix<T>::_correlateFull(const Matrix<T>& kernel, const size_t& resRow,
                            const size_t& resCol) const
{
    T sum = 0;

    for (size_t row = 0; row < rows_; ++row)
    {
        for (size_t col = 0; col < cols_; ++col)
        {
            size_t r = resRow - row;
            size_t c = resCol - col;
            if (resRow >= row && resCol >= col && r < kernel.rows_
                && c < kernel.cols_)
            {
                sum += data_[row][col]
                    * kernel(kernel.rows_ - r - 1, kernel.cols_ - c - 1);
            }
        }
    }

    return sum;
}

// Will only be called on matrix of scalar or it will go kaboom
template <typename T>
T Matrix<T>::_pool(const size_t& poolSize, const size_t& stride,
                   const size_t& resRow, const size_t& resCol)
{
    size_t maxR = resRow * stride;
    size_t maxC = resRow * stride;
    T max = data_[resRow * stride][resCol * stride];

    for (size_t pRow = 0; pRow < poolSize; ++pRow)
    {
        for (size_t pCol = 0; pCol < poolSize; ++pCol)
        {
            size_t r = resRow * stride + pRow;
            size_t c = resCol * stride + pCol;

            T tmp = data_[r][c];
            if (tmp >= max)
            {
                max = tmp;
                maxR = r;
                maxC = c;
            }
            data_[r][c] = 0;
        }
    }
    data_[maxR][maxC] = max;
    return max;
}

template <typename T>
T Matrix<T>::_applyDotProduct(const T& a, const T& b,
                              const MatrixOperation& op) const
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
