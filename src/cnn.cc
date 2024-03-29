#include <iostream>

#include "layer/activation/relu.hh"
#include "layer/convolve_layer.hh"
#include "layer/dense_layer.hh"
#include "layer/flatten_layer.hh"
#include "layer/pooling_layer.hh"
#include "matrix/matrix.hh"

using namespace std;

int main()
{
    Matrix<Matrix<double>> input(2, 1);
    input(0, 0) = Matrix<double>(28, 28);
    input(1, 0) = Matrix<double>(28, 28);

    // Neural neural({ 28, 28, 2 });
    // neural.AddLayer(ConvolveLayer({ 25, 25, 2 }, 32, 2));

    ConvolveLayer convolveLayer1({ 28, 28, 2 }, 32, 3);
    Relu activationLayer1;
    PoolingLayer poolLayer1(2, 2);
    ConvolveLayer convolveLayer2({ 13, 13, 32 }, 64, 3);
    PoolingLayer poolLayer2(2, 2);
    ConvolveLayer convolveLayer3({ 5, 5, 64 }, 64, 3);
    FlattenLayer flattenLayer1({ 3, 3, 64 });
    DenseLayer denseLayer1(576, 64);
    DenseLayer denseLayer2(64, 10);

    // LayerContainer* layers[9] = { &convolveLayer1, &activationLayer1,
    //                               &poolLayer1,     &convolveLayer2,
    //                               &poolLayer2,     &convolveLayer3,
    //                               &flattenLayer1,  &denseLayer1,
    //                               &denseLayer2 };
    // size_t nbr_layers = sizeof(layers) - sizeof(layers[0]);

    Matrix<Matrix<double>> res = convolveLayer1.Forward(input);
    cout << res.Info();
    cout << res(0, 0).Info();

    std::cout << "\nActivation----------\n";
    res = activationLayer1.Forward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    std::cout << "\nPool-----------\n";
    res = poolLayer1.Forward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    std::cout << "\nConvolve-----------\n";
    res = convolveLayer2.Forward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    std::cout << "\nActivation-----------\n";
    res = activationLayer1.Forward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    std::cout << "\nPool-----------\n";
    res = poolLayer2.Forward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    std::cout << "\nConvolve-----------\n";
    res = convolveLayer3.Forward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    std::cout << "\nActivation-----------\n";
    res = activationLayer1.Forward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    std::cout << "\nFlatten-----------\n";
    res = flattenLayer1.Forward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    std::cout << "\nDense-----------\n";
    res = denseLayer1.Forward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    std::cout << "\nDense-----------\n";
    res = denseLayer2.Forward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    cout << "\n\nBackward-----------------------------------\n";

    std::cout << "\nDense----------\n";
    res = denseLayer2.Backward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    std::cout << "\nDense----------\n";
    res = denseLayer1.Backward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    std::cout << "\nFlatten----------\n";
    res = flattenLayer1.Backward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    std::cout << "\nConvolve----------\n";
    res = convolveLayer3.Backward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    std::cout << "\nPool----------\n";
    res = poolLayer2.Backward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    std::cout << "\nConvolve----------\n";
    res = convolveLayer2.Backward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    std::cout << "\nPool----------\n";
    res = poolLayer1.Backward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    std::cout << "\nConvolve----------\n";
    res = convolveLayer1.Backward(res);
    cout << res.Info();
    cout << res(0, 0).Info();

    return 0;
}
