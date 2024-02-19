#pragma once

#include <cstddef>
#include <string>
#include <vector>


class Matrix {
public:
    size_t width;
    size_t height;
    std::vector<std::vector<double>> data; 

public:
    Matrix(size_t height, size_t width);

    Matrix *Add(Matrix *a);
    Matrix *Multiply(Matrix *a);
    Matrix *Transpose(void);
    Matrix *Flip(void);
    Matrix *Convolve(Matrix *a);

    std::string toString();
};
