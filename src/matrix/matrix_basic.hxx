#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <stdexcept>

#include "logger/logger.hh"
#include "matrix/matrix.hh"

/* ---------------Constructors--------------- */
template <typename T>
Matrix<T>::Matrix()
{
    this->rows_ = 0;
    this->cols_ = 0;
}

template <typename T>
Matrix<T>::Matrix(const size_t& rows, const size_t& cols, bool fillRandom)
{
    LOG_INFO("Matrix::constructor with %ld rows and %ld cols", rows, cols);
    this->rows_ = rows;
    this->cols_ = cols;
    data_ = std::vector<std::vector<T>>(rows);
    for (size_t row = 0; row < rows; ++row)
        data_[row] = std::vector<T>(cols);

    if (fillRandom)
        _fillRandomDouble();
}

template <typename T>
Matrix<T>::Matrix(T* values[], const size_t& rows, const size_t& cols,
                  bool fillRandom)
{
    if (!values)
        throw std::invalid_argument("Matrix::Constructor: array is NULL");
    LOG_INFO("Matrix::constructor with %ld rows and %ld cols", rows, cols);

    this->rows_ = rows;
    this->cols_ = cols;
    data_ = std::vector<std::vector<T>>();
    for (size_t row = 0; row < rows; ++row)
    {
        data_.push_back(std::vector<T>(values[row], values[row] + cols));
    }

    if (fillRandom)
        _fillRandomDouble();
}

/* ---------------Getters--------------- */
template <typename T>
size_t Matrix<T>::getRows() const
{
    LOG_INFO("Matrix::getRows on rows:", rows_);
    return rows_;
}

template <typename T>
size_t Matrix<T>::getCols() const
{
    LOG_INFO("Matrix::getCols on cols:", cols_);
    return cols_;
}

template <typename T>
std::vector<std::vector<T>>& Matrix<T>::getData() const
{
    LOG_INFO("Matrix::getData");
    return data_;
}

template <typename T>
const T& Matrix<T>::operator()(const size_t& row, const size_t& col) const
{
    LOG_INFO("Matrix::non-const() at (%ld, %ld) on (%ld, %ld)", row, col, rows_,
             cols_);
    _checkBounds(row, col);
    return data_[row][col];
}

/* ---------------Setters--------------- */
template <typename T>
void Matrix<T>::setRows(const size_t rows)
{
    LOG_INFO("Matrix::setRows to rows:", rows);
    this->rows_ = rows;
}

template <typename T>
void Matrix<T>::setCols(const size_t cols)
{
    LOG_INFO("Matrix::setCols to cols:", cols);
    this->cols_ = cols;
}

template <typename T>
void Matrix<T>::setData(const std::vector<std::vector<T>>& data)
{
    LOG_INFO("Matrix::setData");
    this->data_ = std::move(data);
}

template <typename T>
T& Matrix<T>::operator()(const size_t& row, const size_t& col)
{
    LOG_INFO("Matrix::const() at (%ld, %ld) on (%ld, %ld)", row, col, rows_,
             cols_);
    _checkBounds(row, col);
    return data_[row][col];
}
