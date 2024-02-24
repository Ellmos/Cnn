#include <iostream>

#include "layer/convolve_layer.hpp"
#include "layer/pooling_layer.hpp"
#include "layer/dense_layer.hpp"
#include "layer/flatten_layer.hpp"
#include "matrix/matrix.hpp"

using namespace std;

int main(void)
{
    Matrix<Matrix<double>> input(2, 1);
    input(0, 0) = Matrix<double>(28, 28);
    input(1, 0) = Matrix<double>(28, 28);

    ConvolveLayer convolveLayer1({ 28, 28, 2 }, 32, 3);
    PoolingLayer poolLayer1(2, 2);
    ConvolveLayer convolveLayer2({ 13, 13, 32 }, 64, 3);
    PoolingLayer poolLayer2(2, 2);
    ConvolveLayer convolveLayer3({ 5, 5, 64 }, 64, 3);
    FlattenLayer flattenLayer1({3, 3, 64});
    DenseLayer denseLayer1(576, 64);
    DenseLayer denseLayer2(64, 10);

    Matrix<Matrix<double>> res = convolveLayer1.Forward(input);
    cout << res.Info();
    cout << res(0, 0).Info();

    res = poolLayer1.Forward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    res = convolveLayer2.Forward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    res = poolLayer2.Forward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    res = convolveLayer3.Forward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    Matrix<double> res2 = flattenLayer1.Forward(res);
    cout << res2.Info();

    res2 = denseLayer1.Forward(res2);
    cout << res2.Info();

    res2 = denseLayer2.Forward(res2);
    cout << res2.Info();

    cout << "-----------------------------------" << endl;

    res2 = denseLayer2.Backward(res2);
    cout << res2.Info();

    res2 = denseLayer1.Backward(res2);
    cout << res2.Info();

    res = flattenLayer1.Backward(res2);
    cout << res.Info();
    cout << res(0, 0).Info();

    res = convolveLayer3.Backward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    res = poolLayer2.Backward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    res = convolveLayer2.Backward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    res = poolLayer1.Backward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    res = convolveLayer1.Backward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    return 0;
}
