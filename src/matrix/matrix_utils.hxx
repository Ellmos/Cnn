#pragma once

#include <cstddef>
#include <cstdio>
#include <random>
#include <stdexcept>

#include "matrix/matrix.hh"

template <typename T>
void Matrix<T>::_checkBounds(const size_t& rows, const size_t& cols) const
{
    if (rows >= this->rows_)
    {
        std::string msg = "Matrix::operator(): row index ("
            + std::to_string(rows) + ") out of range";
        throw std::out_of_range(msg);
    }
    if (cols >= this->cols_)
    {
        std::string msg = "Matrix::operator(): column index ("
            + std::to_string(cols) + ") out of range";
        throw std::out_of_range(msg);
    }
}

template <typename T>
void Matrix<T>::_checkDimensionAddition(const Matrix<T>& other,
                                        const std::string& function) const
{
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
        char msg[200];
        std::sprintf(msg,
                     "Matrix::%s: matrices have different sizes\nLHS: "
                     "(%ld, %ld), RHS: (%ld, %ld)",
                     function.c_str(), rows_, cols_, other.rows_, other.cols_);
        throw std::invalid_argument(msg);
    }
}

template <typename T>
void Matrix<T>::_checkDimensionMultiplication(const Matrix<T>& other,
                                              const std::string& function) const
{
    if (cols_ != other.rows_)
    {
        char msg[200];
        std::sprintf(msg,
                     "Matrix::%s: matrices sizes does not match\nLHS: "
                     "(%ld, %ld), RHS: (%ld, %ld)",
                     function.c_str(), rows_, cols_, other.rows_, other.cols_);
        throw std::invalid_argument(msg);
    }
}

template <typename T>
void Matrix<T>::_checkCorrelateArgs(const Matrix<T> kernel,
                                    const std::string& mode,
                                    const std::string& function) const
{
    char msg[200];
    if (kernel.rows_ > rows_ || kernel.cols_ > cols_)
    {
        std::sprintf(msg,
                     "Matrix::%s: kernel size exceeds matrix size\n"
                     "matrix: (%ld, %ld), kernel: (%ld, %ld)",
                     function.c_str(), rows_, cols_, kernel.rows_,
                     kernel.cols_);
        throw std::invalid_argument(msg);
    }
    else if (mode != "valid" && mode != "full")
    {
        std::sprintf(msg, "Matrix::%s: invalid mode given: %s",
                     function.c_str(), mode.data());
        throw std::invalid_argument(msg);
    }
}

template <>
inline void Matrix<double>::_fillRandomDouble()
{
    std::random_device rand;
    std::mt19937 gen(rand());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (size_t i = 0; i < rows_; ++i)
    {
        for (size_t j = 0; j < cols_; ++j)
        {
            data_[i][j] = distribution(gen);
        }
    }
}
template <typename T>
inline void Matrix<T>::_fillRandomDouble()
{}

template <typename T>
void Matrix<T>::_initEmptyMatrix(const Matrix<T>& other)
{
    this->rows_ = other.rows_;
    this->cols_ = other.cols_;
    data_ = std::vector<std::vector<T>>(rows_);
    for (size_t row = 0; row < rows_; ++row)
        data_[row] = std::vector<T>(cols_);
}
