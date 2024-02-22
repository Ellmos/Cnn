#include <iostream>

#include "layer/convolve_layer.hpp"
#include "matrix/matrix.hpp"

using namespace std;

int main(void)
{
    Matrix<Matrix<double>> input(2, 1);
    input(0, 0) = Matrix<double>(28, 28);
    input(1, 0) = Matrix<double>(28, 28);

    // Matrix<Matrix<double>> output(1, 1);
    // output(0, 0) = Matrix<double>(26, 26);
    //
    // Matrix<Matrix<double>> res = input.CustomDotProduct(output, MATRIX_CORRELATE);


    ConvolveLayer layer({ 28, 28, 2 }, 32, 3);
    Matrix<Matrix<double>> res = layer.Forward(input);
    cout << res.Info();
    cout << res(0, 0).Info();


    res = layer.Backward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    return 0;
}
