#pragma once

#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

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
private:
    size_t rows_;
    size_t cols_;
    std::vector<std::vector<T>> data_;

public:
    /* ---------------Basics--------------- */
    Matrix();
    Matrix(const size_t& rows, const size_t& cols, bool fillRandom = true);
    Matrix(T* values[], const size_t& rows, const size_t& cols,
           bool fillRandom = true);

    size_t getRows() const;
    size_t getCols() const;
    std::vector<std::vector<T>>& getData() const;
    const T& operator()(const size_t& row, const size_t& col) const;

    void setRows(const size_t row);
    void setCols(const size_t col);
    void setData(const std::vector<std::vector<T>>& data);
    T& operator()(const size_t& row, const size_t& col);

    /* ---------------Operators--------------- */
    Matrix<T> operator+(const Matrix<T>& other) const;
    Matrix<T> operator-(const Matrix<T>& other) const;
    Matrix<T>& operator+=(const Matrix<T>& other);
    Matrix<T>& operator-=(const Matrix<T>& other);
    Matrix<T> operator*(const Matrix<T>& other) const;
    template <typename U>
    Matrix<T> operator*(const U& val) const;

    /* ---------------Methods--------------- */
    Matrix<T> CustomDotProduct(const Matrix<T>& other,
                               MatrixOperation op) const;
    Matrix<T> Transpose() const;
    Matrix<T> Flip() const;
    Matrix<T> Correlate(const Matrix<T>& kernel, std::string mode) const;
    Matrix<T> Convolve(const Matrix<T>& kernel, std::string mode) const;
    Matrix<T> Pool(const size_t& poolSize, const size_t& stride);
    Matrix<T> Copy() const;
    Matrix<T> Zeros();
    template <typename FUN>
    Matrix<T> Map(FUN fun);
    std::string ToString() const;
    std::string Info() const;

private:
    T _correlateValid(const Matrix<T>& kernel, const size_t& resRow,
                      const size_t& resCol) const;
    T _correlateFull(const Matrix<T>& kernel, const size_t& resRow,
                     const size_t& resCol) const;
    // Will only be called on matrix of scalar or it will go kaboom
    T _pool(const size_t& poolSize, const size_t& stride, const size_t& resRow,
            const size_t& resCol);

    T _applyDotProduct(const T& a, const T& b, const MatrixOperation& op) const;

    /* ---------------Utils--------------- */
    void _checkBounds(const size_t& row, const size_t& col) const;
    void _checkDimensionAddition(const Matrix<T>& other,
                                 const std::string& function) const;
    void _checkDimensionMultiplication(const Matrix<T>& other,
                                       const std::string& function) const;
    void _checkCorrelateArgs(const Matrix<T> kernel, const std::string& mode,
                             const std::string& function) const;
    void _fillRandomDouble();
    void _initEmptyMatrix(const Matrix<T>& other);
};

template <typename T>
struct is_matrix
{
    static const bool value = false;
};

template <typename T>
struct is_matrix<Matrix<T>>
{
    static const bool value = true;
};

#include "matrix_basic.hxx"
#include "matrix_methods.hxx"
#include "matrix_operators.hxx"
#include "matrix_utils.hxx"
