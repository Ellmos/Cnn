#pragma once

#include <cstddef>
#include <cstdio>

#include "logger/logger.hh"
#include "matrix/matrix.hh"

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const
{
    LOG_TRACE("Matrix::operator+ between (%ld, %ld) and (%ld, %ld)", rows_,
              cols_, other.rows_, other.cols_);
    _checkDimensionAddition(other, "operator+");

    Matrix<T> result(rows_, cols_);
    for (size_t row = 0; row < rows_; ++row)
        for (size_t col = 0; col < cols_; ++col)
            result(row, col) = data_[row][col] + other(row, col);
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const
{
    LOG_TRACE("Matrix::operator+ between (%ld, %ld) and (%ld, %ld)", rows_,
              cols_, other.rows_, other.cols_);
    _checkDimensionAddition(other, "operator-");

    Matrix<T> result(rows_, cols_);
    for (size_t row = 0; row < rows_; ++row)
        for (size_t col = 0; col < cols_; ++col)
            result(row, col) = data_[row][col] - other(row, col);
    return result;
}

template <typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& other)
{
    LOG_TRACE("Matrix::operator+= between (%ld, %ld) and (%ld, %ld)", rows_,
              cols_, other.rows_, other.cols_);

    if (rows_ == 0 || cols_ == 0)
        _initEmptyMatrix(other);
    _checkDimensionAddition(other, "operator+=");

    for (size_t i = 0; i < rows_; ++i)
        for (size_t j = 0; j < cols_; ++j)
            data_[i][j] += other(i, j);

    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& other)
{
    LOG_TRACE("Matrix::operator-= between (%ld, %ld) and (%ld, %ld)", rows_,
              cols_, other.rows_, other.cols_);

    if (rows_ == 0 || cols_ == 0)
        _initEmptyMatrix(other);
    _checkDimensionAddition(other, "operator-=");

    for (size_t i = 0; i < rows_; ++i)
        for (size_t j = 0; j < cols_; ++j)
            data_[i][j] -= other(i, j);

    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const
{
    LOG_TRACE("Matrix::operator* between (%ld, %ld) and (%ld, %ld)", rows_,
              cols_, other.rows_, other.cols_);
    _checkDimensionMultiplication(other, "operator*");

    Matrix<T> res(rows_, other.cols_);
    for (size_t i = 0; i < rows_; i++)
    {
        for (size_t j = 0; j < other.cols_; j++)
        {
            T tmp;
            if constexpr (is_matrix<T>::value)
                tmp = T(data_[0][0].rows, data_[0][0].cols);
            else
                tmp = 0;

            for (size_t k = 0; k < cols_; k++)
            {
                tmp += data_[i][k] * other(k, j);
            }
            res(i, j) = tmp;
        }
    }

    return res;
}

template <typename T>
template <typename U>
Matrix<T> Matrix<T>::operator*(const U& val) const
{
    LOG_TRACE("Matrix::operator* between (%ld, %ld) and U", rows_, cols_);

    Matrix<T> result(rows_, cols_);
    for (size_t row = 0; row < rows_; ++row)
        for (size_t col = 0; col < cols_; ++col)
            result(row, col) = data_[row][col] * val;
    return result;
}
