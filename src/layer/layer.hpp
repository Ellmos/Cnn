#pragma once

class Layer
{
public:
    virtual void *Forward(void *input) = 0;
    virtual void *Backward(void *output) = 0;
};
