#include <iostream>

#include "layer/convolve_layer.hpp"

using namespace std;

int main(void)
{
    Matrix<Matrix<double>> input(2, 1);
    input(0, 0) = Matrix<double>(28, 28);
    input(1, 0) = Matrix<double>(28, 28);

    ConvolveLayer layer({ 28, 28, 2 }, 1, 3);
    Matrix<Matrix<double>> res = layer.Forward(input);

    cout << res.ToString();

    return 0;
}
