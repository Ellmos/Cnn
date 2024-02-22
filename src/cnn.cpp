#include <iostream>
#include "layer/convolve_layer.hpp"

using namespace std;

int main(void)
{
    Matrix<Matrix<double>> input(1, 1); 
    input(0, 0) = Matrix<double>(28, 28);


    ConvolveLayer layer({ 28, 28, 1 }, 32, 3);
    layer.Forward(&input);

    cout << layer.biases.Info();
    return 0;
}
