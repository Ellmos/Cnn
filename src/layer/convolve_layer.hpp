#include <vector>

#include "layer.hpp"
#include "matrix/matrix.hpp"

class ConvolveLayer : public Layer
{
public:
    std::vector<Matrix<Matrix<double>> *> kernels;

public:
    void *Forward(void *input) override;
    void *Backward(void *output) override;
};
