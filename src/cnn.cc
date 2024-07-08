#include <iostream>
#include <memory>

#include "layer/activation/relu.hh"
#include "layer/convolve_layer.hh"
#include "layer/dense_layer.hh"
#include "layer/flatten_layer.hh"
#include "layer/pooling_layer.hh"
#include "matrix/matrix.hh"
#include "neural/neural.hh"

int main()
{
    // Neural neural({ 28, 28, 2 });
    // neural.AddLayer(
    //     std::make_unique<ConvolveLayer>(ConvolveLayer({ 28, 28, 2 }, 32,
    //     3)));
    // neural.AddLayer(std::make_unique<Relu>(Relu()));
    // neural.AddLayer(std::make_unique<PoolingLayer>(PoolingLayer(2, 2)));
    // neural.AddLayer(
    //     std::make_unique<ConvolveLayer>(ConvolveLayer({ 13, 13, 32 }, 64,
    //     3)));
    // neural.AddLayer(std::make_unique<Relu>(Relu()));
    // neural.AddLayer(std::make_unique<PoolingLayer>(PoolingLayer(2, 2)));
    // neural.AddLayer(
    //     std::make_unique<ConvolveLayer>(ConvolveLayer({ 5, 5, 64 }, 64, 3)));
    // neural.AddLayer(std::make_unique<FlattenLayer>(FlattenLayer({ 3, 3, 64
    // }))); neural.AddLayer(std::make_unique<DenseLayer>(DenseLayer(576, 64)));
    // neural.AddLayer(std::make_unique<DenseLayer>(DenseLayer(64, 10)));
    //
    Neural neural({ 28, 28, 1 });
    neural.AddLayer(
        std::make_unique<FlattenLayer>(FlattenLayer({ 28, 28, 1 })));
    neural.AddLayer(std::make_unique<DenseLayer>(DenseLayer(784, 32)));
    neural.AddLayer(std::make_unique<DenseLayer>(DenseLayer(32, 10)));

    Matrix<Matrix<double>> input(1, 1);
    input(0, 0) = Matrix<double>(28, 28);
    // input(1, 0) = Matrix<double>(28, 28);

    for (size_t i = 0; i < 600; ++i)
    {
        auto output = neural.Forward(input);
        neural.Backward(output);
        if (i % 100 == 0)
            std::cout << i << "\n";
    }

    return 0;
}
